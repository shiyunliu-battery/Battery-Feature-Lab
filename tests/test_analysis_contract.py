"""End-to-end tests for the compact-index and retrievable-evidence contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import bds
import numpy as np
import polars as pl
import pytest

import bfl
from battery_feature_lab.analysis import adapters, compiler, response
from battery_feature_lab.analysis.schema import (
    ANALYSIS_POLICY_VERSION,
    DEFAULT_ANALYSIS_POLICY,
    METADATA_SCHEMA,
    SUMMARY_SCHEMA,
    VALIDATION_SCHEMA,
    AnalysisConfig,
    metric,
)
from battery_feature_lab.analysis.writer import canonical_sha256, validate_payload


def test_analyze_writes_index_metadata_evidence_and_calls_pyprobe(
    vendor_csv: Path, tmp_path: Path, monkeypatch
) -> None:
    calls = {"cycling": 0, "differentiation": 0}
    original_cycling = response.cycling.summary
    original_differentiation = response.differentiation.differentiate_lean

    def cycling_spy(*args, **kwargs):
        calls["cycling"] += 1
        return original_cycling(*args, **kwargs)

    def differentiation_spy(*args, **kwargs):
        calls["differentiation"] += 1
        return original_differentiation(*args, **kwargs)

    monkeypatch.setattr(response.cycling, "summary", cycling_spy)
    monkeypatch.setattr(response.differentiation, "differentiate_lean", differentiation_spy)
    result = bfl.analyze(vendor_csv, tmp_path / "out", nominal_capacity_ah=1.0)

    assert [path.name for path in result.files] == [
        "normalized_data.bdf.parquet",
        "bds_conversion_report.json",
        "analysis_metadata.json",
        "analysis_results.json",
        "analysis_evidence.json",
        "analysis_validation.json",
    ]
    assert {path.name for path in result.output_dir.iterdir()} == {
        path.name for path in result.files
    }
    assert calls["cycling"] == 10
    assert calls["differentiation"] >= 2
    results = json.loads(result.analysis_results_path.read_text(encoding="utf-8"))
    evidence = json.loads(result.analysis_evidence_path.read_text(encoding="utf-8"))
    metadata = json.loads(result.analysis_metadata_path.read_text(encoding="utf-8"))
    validation = json.loads(result.analysis_validation_path.read_text(encoding="utf-8"))
    assert results["schema_version"] == "bfl.summary/0.1"
    assert evidence["schema_version"] == "bfl.analysis/0.1"
    assert metadata["schema_version"] == "bfl.metadata/0.1"
    assert results["metadata_file"] == result.analysis_metadata_path.name
    assert results["evidence_file"] == result.analysis_evidence_path.name
    assert results["validation_file"] == result.analysis_validation_path.name
    assert "records" not in results
    assert (
        result.analysis_results_path.stat().st_size < result.analysis_evidence_path.stat().st_size
    )
    assert validation["schema_validation"]["analysis_results"]["valid"] is True
    assert validation["recomputation"]["row_count_preserved"] is True
    assert (
        validation["recomputation"]["short_window_recomputation"][
            "relaxation_checkpoint_identities_checked"
        ]
        > 0
    )
    assert len(validation["artifacts"]) == 6
    assert all(item["sha256"] for item in validation["artifacts"].values())
    for path in result.files[:-1]:
        assert (
            validation["artifacts"][path.name]["sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    self_digest = validation["artifacts"]["analysis_validation.json"]["sha256"]
    canonical_scope = json.loads(json.dumps(validation))
    canonical_scope["artifacts"]["analysis_validation.json"]["sha256"] = None
    assert self_digest == canonical_sha256(canonical_scope)
    assert "evolution.capacity" in validation["record_counts"]
    assert validation["record_counts"]["operation.window_summary"] == 1
    assert validation["record_counts"]["response.directional_energy_summary"] == 1
    assert validation["record_counts"]["response.relaxation_signature"] == 1
    assert validation["record_counts"]["response.current_step_summary"] == 1
    modes = {
        item["attributes"]["mode"]
        for item in evidence["records"]
        if item["record_type"] == "operation.mode_segment"
    }
    assert "constant_current_like" in modes
    cycle_records = [
        item for item in evidence["records"] if item["record_type"] == "response.cycle_summary"
    ]
    assert any(item["metrics"]["energy_efficiency"]["status"] == "ok" for item in cycle_records)
    diagnostic_records = [
        item
        for item in evidence["records"]
        if item["record_type"] in {"response.ica_curve", "response.dva_curve"}
        and item["applicability"]["status"] == "applicable"
    ]
    assert diagnostic_records
    assert all(item["series"]["x"] and item["series"]["y"] for item in diagnostic_records)
    assert all("peaks" in item["attributes"] for item in diagnostic_records)
    indexed_ids = {
        item["record_id"] for records in results["dimensions"].values() for item in records
    }
    evidence_ids = {item["record_id"] for item in evidence["records"]}
    assert indexed_ids <= evidence_ids
    assert sum(results["retrieval"]["record_type_counts"].values()) == len(evidence["records"])
    window_index = next(
        item
        for item in results["dimensions"]["operation"]
        if item["record_type"] == "operation.window_summary"
    )
    sequence_status = window_index["attributes"]["operation_sequence_status"]
    assert sequence_status in {"included", "retrieve_segment_records"}
    if sequence_status == "included":
        assert window_index["attributes"]["operation_sequence"]
        assert all(
            step["phase_record_id"] in evidence_ids and step["mode_record_id"] in evidence_ids
            for step in window_index["attributes"]["operation_sequence"]
        )
    else:
        assert window_index["attributes"]["operation_sequence"] == []
        assert results["retrieval"]["record_type_counts"]["operation.phase_segment"] > 32
    assert all(
        item["attributes"]["metadata_context"]["file"] == "analysis_metadata.json"
        for item in evidence["records"]
    )
    assert metadata["cell"]["chemistry"]["status"] == "unknown"

    invalid_summary = json.loads(json.dumps(results))
    invalid_summary["dimensions"]["operation"] = [{}]
    assert validate_payload(invalid_summary, SUMMARY_SCHEMA)
    invalid_metadata = json.loads(json.dumps(metadata))
    invalid_metadata["channels"] = [{}]
    assert validate_payload(invalid_metadata, METADATA_SCHEMA)
    invalid_validation = json.loads(json.dumps(validation))
    invalid_validation["status"] = "garbage"
    assert validate_payload(invalid_validation, VALIDATION_SCHEMA)


def test_bds_report_is_direct_serialization(vendor_csv: Path, tmp_path: Path, monkeypatch) -> None:
    captured = {}
    original = bds.convert

    def convert_spy(*args, **kwargs):
        report = original(*args, **kwargs)
        captured["report"] = report
        return report

    monkeypatch.setattr(bds, "convert", convert_spy)
    result = bfl.analyze(vendor_csv, tmp_path / "out", nominal_capacity_ah=1.0)
    on_disk = json.loads(result.bds_conversion_report_path.read_text(encoding="utf-8"))
    assert on_disk == captured["report"].to_dict()
    normalized = pl.read_parquet(result.normalized_data_path)
    assert "Test Time / s" in normalized.columns
    assert 5e-5 in normalized["Current / A"].to_list()


def test_provider_error_does_not_use_local_cycle_summary_fallback(
    vendor_csv: Path, tmp_path: Path, monkeypatch
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("forced provider failure")

    monkeypatch.setattr(response.cycling, "summary", fail)
    result = bfl.analyze(vendor_csv, tmp_path / "out", nominal_capacity_ah=1.0)
    cycle_records = [
        item for item in result.records if item["record_type"] == "response.cycle_summary"
    ]
    assert cycle_records
    assert all("provider_error" in item["quality"]["flags"] for item in cycle_records)
    assert all(
        item["metrics"]["charge_capacity"]["status"] == "not_computable" for item in cycle_records
    )


def test_public_api_is_exact() -> None:
    assert bfl.__all__ == ["AnalysisResult", "analyze"]


def test_non_finite_metric_is_explicitly_not_computable() -> None:
    value = metric(float("nan"), "V")

    assert value == {
        "value": None,
        "unit": "V",
        "status": "not_computable",
        "reason": "non_finite_value",
    }


def test_schema_validation_rejects_non_finite_numbers() -> None:
    errors = validate_payload({"value": np.nan}, {"type": "object"})

    assert errors == ["$/value: non-finite number nan is not valid JSON"]


def test_time_current_partial_handoff_keeps_current_analysis_and_skips_pyprobe(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "current_only.csv"
    source.write_text(
        "Test Time (s),Current (A)\n0,0\n1,1\n2,-1\n3,0\n",
        encoding="utf-8",
    )

    def fail_bridge(*args, **kwargs):
        raise AssertionError("PyProBE bridge must not be built without voltage")

    monkeypatch.setattr(compiler, "prepare_pyprobe_frame", fail_bridge)
    result = bfl.analyze(source, tmp_path / "out")
    validation = json.loads(result.analysis_validation_path.read_text(encoding="utf-8"))
    metadata = json.loads(result.analysis_metadata_path.read_text(encoding="utf-8"))

    window = next(
        record for record in result.records if record["record_type"] == "operation.window_summary"
    )
    exposure = next(
        record for record in result.records if record["record_type"] == "operation.exposure_summary"
    )
    assert window["applicability"]["status"] == "applicable"
    assert exposure["applicability"]["status"] == "applicable"
    assert window["metrics"]["capacity_throughput"]["value"] == pytest.approx(2 / 3600)
    assert window["metrics"]["voltage_q50"] == {
        "value": None,
        "unit": "V",
        "status": "not_computable",
        "reason": "missing_required_channel:voltage_v",
    }
    assert validation["missing_inputs"] == []
    assert validation["channel_capabilities"]["voltage"]["usable"] is False
    assert all(
        call["status"] == "not_invoked"
        for call in validation["provider_calls"]
        if call["provider"] == "PyProBE"
    )
    assert metadata["dataset"]["conformance_status"] == "bds_partial_legacy_bdf_style"
    assert "Voltage / V" not in pl.read_parquet(result.normalized_data_path).columns
    dependent = [
        record for record in result.records if record["record_type"].startswith("response.")
    ]
    assert dependent
    assert all(record["applicability"]["status"] == "not_computable" for record in dependent)


def test_adding_voltage_enriches_results_without_changing_current_metrics(tmp_path: Path) -> None:
    current_only = tmp_path / "current.csv"
    enriched = tmp_path / "enriched.csv"
    current_only.write_text(
        "Test Time (s),Current (A)\n0,0\n1,1\n2,-1\n3,0\n",
        encoding="utf-8",
    )
    enriched.write_text(
        "Test Time (s),Current (A),Voltage (V)\n0,0,3.0\n1,1,3.1\n2,-1,3.2\n3,0,3.1\n",
        encoding="utf-8",
    )
    partial = bfl.analyze(current_only, tmp_path / "partial")
    rich = bfl.analyze(enriched, tmp_path / "rich")

    names = (
        "charge_throughput",
        "discharge_throughput",
        "capacity_throughput",
        "absolute_current_q50",
        "absolute_current_q95",
        "current_squared_exposure",
    )
    partial_window = next(
        record for record in partial.records if record["record_type"] == "operation.window_summary"
    )
    rich_window = next(
        record for record in rich.records if record["record_type"] == "operation.window_summary"
    )
    assert {name: partial_window["metrics"][name] for name in names} == {
        name: rich_window["metrics"][name] for name in names
    }
    assert partial_window["metrics"]["energy_throughput"]["status"] == "not_computable"
    assert rich_window["metrics"]["energy_throughput"]["status"] == "ok"


def test_output_directory_reuse_refuses_unrelated_files(vendor_csv: Path, tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "unrelated.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="outside this run contract"):
        bfl.analyze(vendor_csv, output)

    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "user data"


def test_analysis_policy_is_validated_and_serialized(tmp_path: Path) -> None:
    source = tmp_path / "current.csv"
    source.write_text(
        "Test Time (s),Current (A)\n0,0\n1,1\n2,-1\n3,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown analysis_policy keys"):
        AnalysisConfig(analysis_policy={"example_specific_threshold": 1.0})

    result = bfl.analyze(
        source,
        tmp_path / "out",
        analysis_policy={"mode_cc_current_cv_max": 0.2},
        relaxation_checkpoints_s=(2.5, 60.0),
    )
    evidence = json.loads(result.analysis_evidence_path.read_text(encoding="utf-8"))
    configuration = evidence["configuration"]

    assert configuration["analysis_policy_version"] == ANALYSIS_POLICY_VERSION
    assert configuration["analysis_policy"]["mode_cc_current_cv_max"] == 0.2
    assert set(configuration["analysis_policy"]) == set(DEFAULT_ANALYSIS_POLICY)
    assert configuration["relaxation_checkpoints_s"] == [2.5, 60.0]


def test_run_id_is_independent_of_output_location(tmp_path: Path) -> None:
    source = tmp_path / "current.csv"
    source.write_text(
        "Test Time (s),Current (A)\n0,0\n1,1\n2,-1\n3,0\n",
        encoding="utf-8",
    )

    first = bfl.analyze(source, tmp_path / "first")
    second = bfl.analyze(source, tmp_path / "second")
    first_evidence = json.loads(first.analysis_evidence_path.read_text(encoding="utf-8"))
    second_evidence = json.loads(second.analysis_evidence_path.read_text(encoding="utf-8"))

    assert first_evidence["configuration"]["output_dir"] != second_evidence["configuration"][
        "output_dir"
    ]
    assert first_evidence["run_id"] == second_evidence["run_id"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nominal_capacity_ah": float("nan")},
        {"pulse_resistance_times_s": (float("inf"),)},
        {"relaxation_checkpoints_s": (float("nan"),)},
        {"rest_current_threshold_a": float("nan")},
        {"sampling_interval_outlier_factor": float("inf")},
    ],
)
def test_nonfinite_public_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="finite"):
        AnalysisConfig(**kwargs)


def test_native_bdf_path_writes_native_report_and_never_calls_bds(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "cell.bdf.parquet"
    pl.DataFrame(
        {
            "Test Time / s": [0.0, 1.0, 2.0, 3.0],
            "Voltage / V": [3.0, 3.1, 3.2, 3.1],
            "Current / A": [0.0, 1.0, -1.0, 0.0],
        }
    ).write_parquet(source)
    sidecar = source.with_name(f"{source.name}.metadata.json")
    sidecar.write_text('{"dataset": "declared elsewhere"}', encoding="utf-8")

    def read(path, *, normalize, validate):
        assert normalize is True and validate is True
        return pl.read_parquet(path)

    def validate(table, *, report, raise_on_error):
        assert report is False and raise_on_error is False
        return {"ok": True, "missing": (), "issues": []}

    def save(table, path, *, labels):
        assert labels == "preferred"
        pl.DataFrame(table).write_parquet(path)

    fake_bdf = SimpleNamespace(
        __version__="0.1.0",
        read=read,
        validate=validate,
        save=save,
    )
    real_import = adapters.importlib.import_module
    monkeypatch.setattr(
        adapters.importlib,
        "import_module",
        lambda name: fake_bdf if name == "bdf" else real_import(name),
    )

    def fail_bds(*args, **kwargs):
        raise AssertionError("native BDF must not be routed through BDS")

    monkeypatch.setattr(bds, "convert", fail_bds)
    result = bfl.analyze(source, tmp_path / "out")
    metadata = json.loads(result.analysis_metadata_path.read_text(encoding="utf-8"))
    validation_payload = json.loads(result.analysis_validation_path.read_text(encoding="utf-8"))

    assert [path.name for path in result.files] == [
        "normalized_data.bdf.parquet",
        "bdf_validation_report.json",
        "analysis_metadata.json",
        "analysis_results.json",
        "analysis_evidence.json",
        "analysis_validation.json",
    ]
    assert result.bds_conversion_report_path is None
    assert result.bdf_validation_report_path == result.input_report_path
    assert metadata["dataset"]["provider"] == "batterydf"
    assert metadata["dataset"]["conformance_status"] == "batterydf_validator_passed"
    assert metadata["test"]["cycler"]["status"] == "unknown"
    assert metadata["semantics"]["current_sign"] == "charge-positive"
    assert metadata["dataset"]["source_artifacts"] == [
        {
            "filename": sidecar.name,
            "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            "role": "bdf_metadata_sidecar",
        }
    ]
    assert validation_payload["provider_calls"][0]["provider"] == "batterydf"
    assert validation_payload["recomputation"]["bdf_validator_passed"] is True
