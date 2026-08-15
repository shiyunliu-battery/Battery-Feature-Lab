"""Tests for the narrow, lossless BDS-to-PyProBE bridge."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from battery_feature_lab.analysis import adapters
from battery_feature_lab.analysis.adapters import (
    analysis_capacity,
    apply_analysis_channel_overrides,
    convert_with_bdf,
    cycle_source,
    detect_input_adapter,
    prepare_analysis_frame,
    prepare_pyprobe_frame,
)


def test_bridge_preserves_cycle_step_source_mapping_and_only_zeroes_staging_current() -> None:
    canonical = pl.DataFrame(
        {
            "_source_row": [8, 9, 10, 11],
            "test_time_s": [0.0, 1.0, 3.0, 4.0],
            "voltage_v": [3.0, 3.1, 3.2, 3.3],
            "current_a": [5e-5, 1.0, 1.0, 0.0],
            "cycle_index": [4, 4, 4, 4],
            "step_index": [7, 7, 8, 8],
            "record_index": [101, 102, 103, 104],
        }
    )

    staged, details = prepare_pyprobe_frame(
        canonical,
        rest_current_threshold_a=1e-4,
        sampling_interval_outlier_factor=5.0,
        reported_capacity_finite_coverage_min=0.99,
    )

    assert canonical["current_a"][0] == 5e-5
    assert staged["Current [A]"][0] == 0.0
    assert staged["Cycle"].to_list() == [4, 4, 4, 4]
    assert staged["Step"].to_list() == [7, 7, 8, 8]
    assert staged["Event"].to_list() == [0, 0, 1, 1]
    assert staged["_source_row"].to_list() == [8, 9, 10, 11]
    assert staged["_source_record"].to_list() == [101, 102, 103, 104]
    expected = np.cumsum([0.0, 5e-5 / 3600, 2 / 3600, 1 / 3600])
    np.testing.assert_allclose(staged["Capacity [Ah]"], expected)
    assert details["capacity_method"] == "zoh_previous_v1"


def test_reported_capacity_is_validated_within_one_cycle() -> None:
    frame = pl.DataFrame(
        {
            "test_time_s": [0.0, 1.0, 2.0, 3.0],
            "current_a": [1.0, 1.0, -1.0, -1.0],
            "charge_capacity_ah": [0.0, 0.1, 0.2, 0.2],
            "discharge_capacity_ah": [0.0, 0.0, 0.0, 0.1],
        }
    )

    capacity, details = analysis_capacity(
        frame,
        sampling_interval_outlier_factor=5.0,
        reported_capacity_finite_coverage_min=0.99,
    )

    np.testing.assert_allclose(capacity, [0.0, 0.1, 0.2, 0.1])
    assert details == {
        "capacity_method": "reported_charge_minus_discharge",
        "capacity_flags": [],
        "reported_capacity_finite_coverage_min": 0.99,
    }


def test_analysis_frame_excludes_only_rows_without_finite_time() -> None:
    frame = pl.DataFrame(
        {
            "_source_row": [4, 5, 6, 7],
            "test_time_s": [2.0, None, 1.0, float("nan")],
            "current_a": [1.0, None, None, 2.0],
            "voltage_v": [3.2, None, 3.1, 3.3],
        }
    )

    analysis_frame, details = prepare_analysis_frame(frame)

    assert analysis_frame["_source_row"].to_list() == [6, 4]
    assert analysis_frame["current_a"].to_list() == [None, 1.0]
    assert details == {
        "normalized_row_count": 4,
        "analysis_row_count": 2,
        "excluded_rows_without_finite_time": 2,
        "excluded_source_rows_without_finite_time": [5, 7],
    }


def test_explicit_raw_temperature_channel_is_audited_analysis_alias() -> None:
    frame = pl.DataFrame(
        {
            "voltage_v": [3.0, 3.1],
            "raw:Surface_Temp(degC)": [5.0, 5.2],
        }
    )

    selected, details = apply_analysis_channel_overrides(
        frame,
        voltage_column=None,
        temperature_column_name="raw:Surface_Temp(degC)",
    )

    assert "analysis_temperature_deg_c" not in frame.columns
    assert selected["analysis_temperature_deg_c"].to_list() == [5.0, 5.2]
    assert details["temperature_column_source"] == "explicit"
    assert details["temperature_finite_sample_fraction"] == 1.0
    assert details["bdf_file_modified"] is False


@pytest.mark.parametrize(
    "columns",
    [
        {"Test Time / s": [0.0, 1.0], "Voltage / V": [3.0, 3.1], "Current / A": [1.0, 1.0]},
        {
            "test_time_second": [0.0, 1.0],
            "voltage_volt": [3.0, 3.1],
            "current_ampere": [1.0, 1.0],
        },
    ],
)
def test_formal_bdf_adapter_uses_provider_and_maps_label_variants(
    columns: dict[str, list[float]], tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "input.bdf.parquet"
    pl.DataFrame(columns).write_parquet(source)

    def read(path, *, normalize, validate):
        assert normalize is True and validate is True
        return pl.read_parquet(path)

    def validate(table, *, report, raise_on_error):
        assert report is False and raise_on_error is False
        return {"ok": True, "missing": (), "issues": []}

    def save(table, path, *, labels):
        assert labels == "preferred"
        pl.DataFrame(table).write_parquet(path)

    fake = SimpleNamespace(
        __version__="0.1.0",
        read=read,
        validate=validate,
        save=save,
    )
    real_import = adapters.importlib.import_module

    def import_module(name: str):
        return fake if name == "bdf" else real_import(name)

    monkeypatch.setattr(adapters.importlib, "import_module", import_module)
    normalized = tmp_path / "normalized_data.bdf.parquet"
    report_path = tmp_path / "bdf_validation_report.json"

    frame, report, metadata, version = convert_with_bdf(source, normalized, report_path)

    assert {"test_time_s", "voltage_v", "current_a", "_source_row"} <= set(frame.columns)
    assert report == json.loads(report_path.read_text(encoding="utf-8"))
    assert metadata["provider_api_variant"] == "pypi_0.1_dataframe"
    assert version == "0.1.0"


def test_formal_bdf_provider_failure_never_falls_back_to_bds(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "invalid.bdf.parquet"
    pl.DataFrame({"Current / A": [1.0]}).write_parquet(source)
    fake = SimpleNamespace(
        __version__="0.1.0",
        read=lambda *args, **kwargs: pl.read_parquet(source),
        validate=lambda *args, **kwargs: {
            "ok": False,
            "detail": "missing Test Time / s and Voltage / V",
        },
    )
    real_import = adapters.importlib.import_module
    monkeypatch.setattr(
        adapters.importlib,
        "import_module",
        lambda name: fake if name == "bdf" else real_import(name),
    )

    with pytest.raises(ValueError, match="formal BDF validation failed"):
        convert_with_bdf(
            source,
            tmp_path / "normalized_data.bdf.parquet",
            tmp_path / "bdf_validation_report.json",
        )


def test_auto_routes_explicit_and_plain_canonical_bdf(tmp_path: Path) -> None:
    explicit = tmp_path / "data.bdf.csv"
    explicit.write_text("not,inspected\n", encoding="utf-8")
    plain = tmp_path / "data.csv"
    plain.write_text(
        "Test Time / s,Voltage / V,Current / A\n0,3.0,1.0\n",
        encoding="utf-8",
    )
    vendor = tmp_path / "vendor.csv"
    vendor.write_text("Time,Volts,Amps\n0,3.0,1.0\n", encoding="utf-8")

    assert detect_input_adapter(explicit) == "bdf"
    assert detect_input_adapter(plain) == "bdf"
    assert detect_input_adapter(vendor) == "bds"


def test_auto_preserves_bfl_handoff_identity_from_validation(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized_data.bdf.parquet"
    pl.DataFrame(
        {"Test Time / s": [0.0], "Voltage / V": [3.0], "Current / A": [0.0]}
    ).write_parquet(normalized)
    (tmp_path / "analysis_validation.json").write_text(
        json.dumps({"recomputation": {"input_adapter": "bds"}}),
        encoding="utf-8",
    )

    assert detect_input_adapter(normalized) == "bds"


def test_unknown_cycle_origin_is_conservatively_unknown() -> None:
    frame = pl.DataFrame({"cycle_index": [1, 1]})
    report = SimpleNamespace(metadata={}, provenance=[])

    assert cycle_source(frame, report) == "unknown"
