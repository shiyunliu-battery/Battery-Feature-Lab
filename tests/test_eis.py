"""Spectrum provenance, provider boundaries and comparison regression tests."""

from __future__ import annotations

import importlib.util
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import bfl
from battery_feature_lab.analysis.writer import sha256_file
from battery_feature_lab.cli import main
from battery_feature_lab.eis.analysis import eisfit_reference

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("eisfit", "impedance")),
    reason="EIS regression tests require battery-feature-lab[eis]",
)


def source(tmp_path, **extra):
    frequency = np.logspace(4, -1, 32)
    impedance = 0.01 + 0.02 / (1 + 2j * np.pi * frequency * 0.1)
    data = {
        "Frequency_Hz": frequency,
        "Real_Ohm": impedance.real,
        "Imag_Ohm": impedance.imag,
        **extra,
    }
    path = tmp_path / "source.csv"
    pl.DataFrame(data).write_csv(path)
    return path


def observed(tmp_path, path, **options):
    return bfl.analyze_eis(path, tmp_path / "out", run_kk=False, run_fit=False, **options)


def find(result, kind):
    return next(r for r in result.records if r["record_type"] == kind)


def test_source_requires_declared_or_recorded_spectrum_identity(tmp_path):
    with pytest.raises(ValueError, match="Spectrum identity"):
        observed(tmp_path, source(tmp_path))
    assert not (tmp_path / "out").exists()


def test_eis_contract_paths_hashes_and_record_references(tmp_path):
    result = observed(tmp_path, source(tmp_path), single_spectrum=True)
    assert [p.name for p in result.files] == [
        "normalized_eis.parquet",
        "eis_conversion_report.json",
        "analysis_metadata.json",
        "analysis_results.json",
        "analysis_evidence.json",
        "analysis_validation.json",
    ]
    summary = json.loads(result.analysis_results_path.read_text())
    evidence = json.loads(result.analysis_evidence_path.read_text())
    validation = json.loads(result.analysis_validation_path.read_text())
    assert summary["schema_version"] == "bfl.eis-summary/0.2"
    assert evidence["schema_version"] == "bfl.eis-analysis/0.1"
    assert validation["row_accounting"] == {
        "input_rows": 32,
        "retained_rows": 32,
        "excluded_rows": 0,
        "accounted": True,
    }
    for p in result.files[:5]:
        assert validation["artifacts"][p.name]["sha256"] == sha256_file(p)
    ids = {r["record_id"] for r in evidence["records"]}
    assert all(
        r["evidence"]["record_id"] in ids for rs in summary["dimensions"].values() for r in rs
    )
    assert str(tmp_path) not in result.analysis_evidence_path.read_text()
    conditions = find(result, "operation.eis_conditions")["attributes"]["conditions"]
    assert conditions["cell_id"]["value"] is None
    assert conditions["temperature_deg_c"]["status"] == "not_reported"


def test_invalid_points_have_exact_row_accounting(tmp_path):
    path = source(tmp_path)
    frame = pl.read_csv(path).with_columns(
        pl.when(pl.int_range(pl.len()) == 5)
        .then(-1)
        .otherwise(pl.col("Frequency_Hz"))
        .alias("Frequency_Hz")
    )
    frame.write_csv(path)
    result = observed(tmp_path, path, single_spectrum=True)
    report = json.loads(result.input_report_path.read_text())
    assert report["spectra"][0]["excluded_rows"] == [
        {"source_row": 5, "reason": "non_positive_frequency"}
    ]
    normalized = pl.read_parquet(result.normalized_data_path)
    assert normalized.height == 31 and 5 not in normalized["measurement_source_row"].to_list()


def test_all_invalid_spectrum_keeps_unavailable_records(tmp_path):
    path = tmp_path / "bad.csv"
    pl.DataFrame(
        {"Frequency_Hz": [-1.0, 0.0], "Real_Ohm": [1.0, 2.0], "Imag_Ohm": [0.0, 0.0]}
    ).write_csv(path)
    result = observed(tmp_path, path, single_spectrum=True)
    assert find(result, "response.eis_spectrum")["applicability"]["status"] == "not_computable"
    assert pl.read_parquet(result.normalized_data_path).height == 0
    assert json.loads(result.analysis_validation_path.read_text())["row_accounting"]["accounted"]


@pytest.mark.parametrize("unit,phase", [("deg", 5.0), ("rad", np.deg2rad(5.0))])
def test_phase_units_are_explicit_even_for_small_degree_values(tmp_path, unit, phase):
    path = tmp_path / "polar.csv"
    pl.DataFrame(
        {"Frequency_Hz": [1.0, 10.0], "Magnitude_Ohm": [2.0, 2.0], "Phase": [phase, phase]}
    ).write_csv(path)
    result = observed(tmp_path, path, single_spectrum=True, phase_unit=unit)
    normalized = pl.read_parquet(result.normalized_data_path)
    assert normalized["Zim_exp_Ohm"][0] == pytest.approx(2 * np.sin(np.deg2rad(5)))


def test_ambiguous_phase_unit_is_refused(tmp_path):
    path = tmp_path / "polar.csv"
    pl.DataFrame(
        {"Frequency_Hz": [1.0, 10.0], "Magnitude_Ohm": [2.0, 2.0], "Phase": [5.0, 5.0]}
    ).write_csv(path)
    with pytest.raises(ValueError, match="explicit phase_unit"):
        observed(tmp_path, path, single_spectrum=True)


def test_source_phase_unit_cannot_be_overridden_inconsistently(tmp_path):
    path = tmp_path / "polar.csv"
    pl.DataFrame(
        {"Frequency_Hz": [1.0, 10.0], "Magnitude_Ohm": [2.0, 2.0], "Phase_deg": [5.0, 5.0]}
    ).write_csv(path)
    with pytest.raises(ValueError, match="conflicts"):
        observed(tmp_path, path, single_spectrum=True, phase_unit="rad")


def test_negative_imaginary_column_is_converted_once(tmp_path):
    path = tmp_path / "sign.csv"
    pl.DataFrame(
        {"Frequency_Hz": [1.0, 10.0], "Real_Ohm": [1.0, 1.0], "-Im(Z)/Ohm": [0.2, 0.3]}
    ).write_csv(path)
    result = observed(tmp_path, path, single_spectrum=True)
    assert pl.read_parquet(result.normalized_data_path)["Zim_exp_Ohm"].to_list() == [-0.3, -0.2]


def test_multiple_spectra_are_not_fitted_as_one(tmp_path):
    path = source(tmp_path, Spectrum_ID=["a"] * 16 + ["b"] * 16, Cell_ID=["cell"] * 32)
    result = observed(tmp_path, path)
    spectra = [r for r in result.records if r["record_type"] == "response.eis_spectrum"]
    assert len(spectra) == 2
    assert {r["metrics"]["valid_point_count"]["value"] for r in spectra} == {16}
    assert set(spectra[0]["source_intervals"][0]["source_rows"]).isdisjoint(
        spectra[1]["source_intervals"][0]["source_rows"]
    )


def test_varying_conditions_within_one_spectrum_are_refused(tmp_path):
    path = source(tmp_path, Temperature_degC=[25.0] * 16 + [35.0] * 16)
    with pytest.raises(ValueError, match="varies within"):
        observed(tmp_path, path, single_spectrum=True)


def test_duplicate_grid_skips_providers_without_averaging(tmp_path, monkeypatch):
    import eisfit
    from impedance import validation

    def unexpected(*args, **kwargs):
        pytest.fail("Provider must not be called for ambiguous repeated frequencies")

    monkeypatch.setattr(eisfit, "fit_model_eis", unexpected)
    monkeypatch.setattr(validation, "linKK", unexpected)
    path = source(tmp_path)
    frame = pl.read_csv(path)
    pl.concat([frame, frame.head(1)]).write_csv(path)
    result = bfl.analyze_eis(path, tmp_path / "out", single_spectrum=True)
    assert (
        find(result, "response.eis_spectrum")["metrics"]["duplicate_frequency_count"]["value"] == 1
    )
    assert find(result, "response.eis_ecm")["applicability"]["reasons"] == ["duplicate_frequencies"]


def test_probe_interpolation_uses_log_frequency_without_extrapolation(tmp_path):
    path = tmp_path / "linear.csv"
    pl.DataFrame(
        {"Frequency_Hz": [1.0, 10.0], "Real_Ohm": [0.0, 1.0], "Imag_Ohm": [0.0, -1.0]}
    ).write_csv(path)
    query = np.sqrt(10)
    result = observed(tmp_path, path, single_spectrum=True, probe_frequencies_hz=(query, 100.0))
    metrics = find(result, "response.eis_spectrum")["metrics"]
    assert metrics[f"z_real_at_{query:.15g}_hz"]["value"] == pytest.approx(0.5)
    assert metrics["z_real_at_100_hz"]["status"] == "not_computable"


def test_failed_fit_retains_observations_and_optimizer_attempts(tmp_path):
    import eisfit

    f = np.logspace(4, -1, 32)
    p = np.array([1e-7, 0.014, 0.003, 0.7, 0.82, 0.0015, 1.2, 0.9, 700.0, 0.55])
    z = eisfit.model_eis(p, 2 * np.pi * f)
    path = tmp_path / "fit.csv"
    pl.DataFrame({"Frequency_Hz": f, "Real_Ohm": z.real, "Imag_Ohm": z.imag}).write_csv(path)
    result = bfl.analyze_eis(
        path,
        tmp_path / "out",
        single_spectrum=True,
        run_kk=False,
        fit_options={"n_starts": 1, "max_nfev": 1},
    )
    fit = find(result, "response.eis_ecm")
    assert fit["attributes"]["convergence"] == "not_converged"
    assert not fit["metrics"] and not fit["attributes"]["attempts"][0]["accepted"]
    assert find(result, "response.eis_spectrum")["metrics"]["valid_point_count"]["value"] == 32


def test_lin_kk_runs_with_declared_configuration(tmp_path):
    result = bfl.analyze_eis(
        source(tmp_path),
        tmp_path / "out",
        single_spectrum=True,
        run_fit=False,
        kk_residual_limit=0.1,
        kk_options={"max_M": 20, "fit_type": "complex"},
    )
    kk = find(result, "response.eis_kk")
    assert kk["applicability"]["status"] in {"applicable", "partial"}
    assert kk["method"]["parameters"]["max_M"] == 20
    assert kk["metrics"]["maximum_normalized_residual"]["value"] < 0.1
    assert kk["attributes"]["measurement_consistency"] == "within_configured_residual_limit"


def test_undeclared_ecm_parameter_units_are_rejected(tmp_path, monkeypatch):
    import eisfit

    def fitted(f, z, options):
        return SimpleNamespace(
            optimizer=SimpleNamespace(success=True),
            nrmse=0.001,
            rmse=0.0001,
            parameters={"R0_Ohm": 0.01, "tau1": 0.002},
            Zsim=z.copy(),
            bound_hits=[],
            diagnostics={},
        )

    monkeypatch.setattr(eisfit, "fit_model_eis", fitted)
    result = bfl.analyze_eis(source(tmp_path), tmp_path / "out", single_spectrum=True, run_kk=False)
    fit = find(result, "response.eis_ecm")
    assert fit["applicability"]["reasons"] == [
        "provider_invalid_result:ecm_parameter_units_undeclared"
    ]
    assert fit["attributes"]["undeclared_parameters"] == ["tau1"]
    assert not fit["metrics"]


def test_bfl_records_carry_the_installed_package_version(tmp_path):
    result = observed(tmp_path, source(tmp_path), single_spectrum=True)
    spectrum = find(result, "response.eis_spectrum")
    assert spectrum["method"]["provider_version"] == f"BFL {version('battery-feature-lab')}"


def test_lin_kk_locates_the_largest_residual_and_counts_exceedances(tmp_path):
    limit = 1e-12
    result = bfl.analyze_eis(
        source(tmp_path), tmp_path / "out", single_spectrum=True, run_fit=False,
        kk_residual_limit=limit,
    )
    kk = find(result, "response.eis_kk")
    frequency = np.asarray(kk["series"]["frequency_hz"])
    magnitude = np.maximum(
        np.abs(kk["series"]["residual_real"]), np.abs(kk["series"]["residual_imaginary"])
    )
    metrics = kk["metrics"]
    assert metrics["frequency_of_maximum_residual"]["value"] == frequency[np.argmax(magnitude)]
    assert metrics["frequency_of_maximum_residual"]["unit"] == "Hz"
    assert metrics["points_above_residual_limit"]["value"] == int(np.sum(magnitude > limit))
    assert metrics["fraction_points_above_residual_limit"]["value"] == pytest.approx(
        np.sum(magnitude > limit) / len(frequency)
    )


def test_non_finite_lin_kk_mu_is_reported_without_inferring_a_cause(tmp_path, monkeypatch):
    from impedance import validation

    def degenerate(f, z, **options):
        return 1, -np.inf, z, np.zeros(len(f)), np.zeros(len(f))

    monkeypatch.setattr(validation, "linKK", degenerate)
    result = bfl.analyze_eis(
        source(tmp_path), tmp_path / "out", single_spectrum=True, run_fit=False
    )
    kk = find(result, "response.eis_kk")
    assert kk["applicability"]["reasons"] == ["provider_invalid_result:lin_kk_mu_not_finite"]
    assert kk["attributes"]["measurement_consistency"] == "not_assessed"
    assert kk["attributes"]["lin_kk_termination"] == {
        "rc_element_count": 1,
        "mu": None,
        "mu_class": "negative_infinity",
        "invalid_outputs": [],
    }
    assert not kk["metrics"]
    calls = json.loads(result.analysis_validation_path.read_text())["provider_calls"]
    call = next(c for c in calls if c["method"] == "kk")
    assert call["status"] == "error"
    assert call["reason"] == "provider_invalid_result:lin_kk_mu_not_finite"


def test_provider_output_is_counted_without_host_paths(tmp_path, monkeypatch):
    import warnings

    from impedance import validation

    real_lin_kk = validation.linKK

    def noisy(f, z, **options):
        print("progress 1")
        print("progress 2")
        warnings.warn(f"divide by zero in {tmp_path / 'validation.py'}", RuntimeWarning)
        return real_lin_kk(f, z, **options)

    monkeypatch.setattr(validation, "linKK", noisy)
    result = bfl.analyze_eis(
        source(tmp_path), tmp_path / "out", single_spectrum=True, run_fit=False
    )
    text = result.analysis_validation_path.read_text()
    call = next(c for c in json.loads(text)["provider_calls"] if c["method"] == "kk")
    output = call["provider_output"]
    assert output["stdout_line_count"] >= 2
    assert {
        "category": "RuntimeWarning",
        "message": "divide by zero in <path>",
        "count": 1,
    } in output["warnings"]
    assert tmp_path.name not in text


def test_compact_summary_states_shared_conditions_and_limits_once(tmp_path):
    metadata = comparison_metadata()
    metadata["spectra"] = {"a": {"sequence_index": 1}, "b": {"sequence_index": 2}}
    result = observed(
        tmp_path,
        comparison_source(tmp_path),
        metadata=metadata,
        comparison_pairs=(("a", "b"),),
        comparison_kind="ageing",
    )
    summary = json.loads(result.analysis_results_path.read_text())
    evidence = {
        r["record_id"]: r for r in json.loads(result.analysis_evidence_path.read_text())["records"]
    }
    assert summary["conditions"]["shared"]["temperature_deg_c"]["value"] == 25.0
    assert "elapsed_time_s" in summary["conditions"]["not_reported"]
    operations = summary["dimensions"]["operation"]
    assert [list(op["attributes"]["conditions"]) for op in operations] == [["sequence_index"]] * 2
    full_conditions = evidence[operations[0]["record_id"]]["attributes"]["conditions"]
    assert {"temperature_deg_c", "elapsed_time_s", "sequence_index"} <= set(full_conditions)
    comparison = summary["dimensions"]["evolution"][0]
    assert "conditions" not in comparison["attributes"]
    assert [s["spectrum_id"] for s in comparison["attributes"]["compared_spectra"]] == ["a", "b"]
    for record in (r for records in summary["dimensions"].values() for r in records):
        assert "member_summaries" not in record["attributes"]
        assert [
            summary["interpretation_limits"][key] for key in record["interpretation_limits"]
        ] == evidence[record["record_id"]]["interpretation_limits"]


def comparison_source(tmp_path):
    path = source(tmp_path)
    frame = pl.read_csv(path)
    pl.concat(
        [
            frame.with_columns(pl.lit("a").alias("Spectrum_ID")),
            frame.with_columns(pl.lit("b").alias("Spectrum_ID")),
        ]
    ).write_csv(path)
    return path


def comparison_metadata():
    return {
        "defaults": {
            "cell_id": "cell",
            "temperature_deg_c": 25.0,
            "soc_fraction": 0.5,
            "rest_time_s": 3600.0,
            "instrument": "instrument",
            "protocol_id": "protocol",
            "ac_amplitude_v": 0.01,
        }
    }


@pytest.mark.parametrize(
    "change,expected",
    [(None, "comparable"), ("missing", "conditional"), ("temperature", "incompatible")],
)
def test_comparison_gates_match_the_declared_conditions(tmp_path, change, expected):
    metadata = comparison_metadata()
    if change == "missing":
        metadata["defaults"].pop("soc_fraction")
    elif change == "temperature":
        metadata["spectra"] = {"b": {"temperature_deg_c": 35.0}}
    result = observed(
        tmp_path, comparison_source(tmp_path), metadata=metadata, comparison_pairs=(("a", "b"),)
    )
    comparison = find(result, "evolution.eis_comparison")
    assert comparison["attributes"]["comparison_status"] == expected
    if expected == "comparable":
        assert comparison["metrics"]["delta_z_real_at_1_hz"]["value"] == pytest.approx(0)
    else:
        assert not comparison["metrics"]


def test_temperature_comparison_allows_only_the_selected_condition_to_change(tmp_path):
    metadata = comparison_metadata()
    metadata["spectra"] = {"b": {"temperature_deg_c": 35.0}}
    result = observed(
        tmp_path,
        comparison_source(tmp_path),
        metadata=metadata,
        comparison_pairs=(("a", "b"),),
        comparison_kind="temperature",
    )
    assert (
        find(result, "evolution.eis_comparison")["attributes"]["comparison_status"] == "comparable"
    )


def test_ageing_comparison_requires_recorded_order(tmp_path):
    result = observed(
        tmp_path,
        comparison_source(tmp_path),
        metadata=comparison_metadata(),
        comparison_pairs=(("a", "b"),),
        comparison_kind="ageing",
    )
    assert (
        "recorded_ageing_order"
        in find(result, "evolution.eis_comparison")["attributes"]["missing_conditions"]
    )


def test_run_id_does_not_depend_on_output_location(tmp_path):
    path = source(tmp_path)
    first = observed(tmp_path, path, single_spectrum=True)
    second = bfl.analyze_eis(
        path, tmp_path / "second", single_spectrum=True, run_kk=False, run_fit=False
    )
    assert (
        json.loads(first.analysis_results_path.read_text())["run_id"]
        == json.loads(second.analysis_results_path.read_text())["run_id"]
    )


def test_existing_output_is_not_overwritten(tmp_path):
    path = source(tmp_path)
    observed(tmp_path, path, single_spectrum=True)
    with pytest.raises(ValueError, match="must be empty"):
        observed(tmp_path, path, single_spectrum=True)


def test_cli_writes_six_eis_artifacts(tmp_path, capsys):
    code = main(
        [
            "analyze-eis",
            str(source(tmp_path)),
            "--single-spectrum",
            "--no-fit",
            "--no-kk",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0 and len(capsys.readouterr().out.splitlines()) == 6


@pytest.mark.parametrize(
    "options",
    [
        {"phase_unit": "unknown"},
        {"fit_options": {"n_starts": 0}},
        {"kk_options": {"max_M": 1}},
        {"probe_frequencies_hz": (float("nan"),)},
    ],
)
def test_invalid_configuration_fails_before_intake(tmp_path, options):
    with pytest.raises(ValueError):
        bfl.analyze_eis(tmp_path / "absent.csv", **options)


def test_real_spectrum_uses_released_fit_provider(tmp_path):
    path = Path(__file__).parent / "data" / "eis" / "ID01.csv"
    result = bfl.analyze_eis(
        path,
        tmp_path / "out",
        single_spectrum=True,
        metadata={"defaults": {"cell_id": "ID01"}},
        fit_options={"n_starts": 3, "max_nfev": 8000},
    )
    fit = find(result, "response.eis_ecm")
    assert fit["method"]["provider_version"] == version("eisfit")
    assert fit["method"]["references"] == [eisfit_reference()]
    assert fit["attributes"]["convergence"] == "converged"
    assert fit["metrics"]["nrmse"]["value"] < 0.02
    assert fit["attributes"]["identifiability"] == "not_assessed"
    assert (
        find(result, "response.eis_kk")["metrics"]["maximum_normalized_residual"]["value"]
        is not None
    )


def test_metadata_sidecar_is_hashed_without_host_paths(tmp_path):
    metadata = tmp_path / "declared.json"
    metadata.write_text(json.dumps(comparison_metadata()))
    result = observed(tmp_path, source(tmp_path), single_spectrum=True, metadata_path=metadata)
    payload = json.loads(result.analysis_metadata_path.read_text())
    assert payload["source"]["metadata_artifact"] == {
        "filename": metadata.name,
        "sha256": sha256_file(metadata),
    }
    assert str(tmp_path) not in result.analysis_evidence_path.read_text()


def test_failed_lin_kk_does_not_fabricate_measurement_consistency(tmp_path, monkeypatch):
    from impedance import validation

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate provider failure")

    monkeypatch.setattr(validation, "linKK", fail)
    result = bfl.analyze_eis(
        source(tmp_path), tmp_path / "out", single_spectrum=True, run_fit=False
    )
    kk = find(result, "response.eis_kk")
    assert kk["applicability"]["status"] == "not_computable"
    assert kk["attributes"]["measurement_consistency"] == "not_assessed"
    assert not kk["metrics"]
    assert find(result, "response.eis_spectrum")["metrics"]["valid_point_count"]["value"] == 32


def test_noncanonical_units_cannot_be_silently_relabelled(tmp_path):
    path = source(tmp_path)
    frame = pl.read_csv(path).rename({"Frequency_Hz": "Frequency_kHz"})
    frame.write_csv(path)
    with pytest.raises(ValueError, match="require Hz and Ohm"):
        observed(tmp_path, path, single_spectrum=True, columns={"frequency": "Frequency_kHz"})


def test_excel_sheet_identity_is_preserved(tmp_path):
    import pandas as pd

    path = tmp_path / "source.xlsx"
    frame = pl.read_csv(source(tmp_path)).to_pandas()
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="spectrum", index=False)
    result = observed(tmp_path, path, sheet="spectrum", single_spectrum=True)
    assert find(result, "response.eis_spectrum")["source_intervals"][0]["sheet"] == "spectrum"


def test_model_parameter_differences_keep_cpe_unit_and_identity_limits(tmp_path, monkeypatch):
    import eisfit
    from impedance import validation

    def consistent(f, z, **kwargs):
        return 3, 0.5, z.copy(), np.zeros(len(f)), np.zeros(len(f))

    count = 0

    def fitted(f, z, options):
        nonlocal count
        count += 1
        parameters = {
            "L_H": 1e-7,
            "R0_Ohm": count * 0.01,
            "R1_Ohm": 0.01,
            "Q1": 0.5,
            "n1": 0.8 if count == 1 else 0.9,
            "R2_Ohm": 0.02,
            "Q2": 1.0,
            "n2": 0.9,
            "Qd": 1.0,
            "nd": 0.5,
        }
        return SimpleNamespace(
            optimizer=SimpleNamespace(success=True),
            nrmse=0.001,
            rmse=0.0001,
            parameters=parameters,
            Zsim=z.copy(),
            bound_hits=[],
            diagnostics={},
        )

    monkeypatch.setattr(validation, "linKK", consistent)
    monkeypatch.setattr(eisfit, "fit_model_eis", fitted)
    result = bfl.analyze_eis(
        comparison_source(tmp_path),
        tmp_path / "out",
        metadata=comparison_metadata(),
        comparison_pairs=(("a", "b"),),
    )
    comparison = find(result, "evolution.eis_comparison")
    assert comparison["attributes"]["ecm_parameter_comparison_status"] == "conditional"
    assert comparison["metrics"]["delta_model_parameter_R0_Ohm"]["value"] == pytest.approx(0.01)
    assert (
        comparison["metrics"]["delta_model_parameter_Q1"]["reason"]
        == "cpe_exponent_dependent_units_differ"
    )


def test_missing_fit_provider_preserves_observed_evidence(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "eisfit", None)
    result = bfl.analyze_eis(source(tmp_path), tmp_path / "out", single_spectrum=True, run_kk=False)
    fit = find(result, "response.eis_ecm")
    assert fit["applicability"]["status"] == "not_computable"
    assert "provider_unavailable" in fit["applicability"]["reasons"][0]
    assert find(result, "response.eis_spectrum")["metrics"]["valid_point_count"]["value"] == 32


def test_metadata_typo_cannot_be_silently_ignored(tmp_path):
    with pytest.raises(ValueError, match="absent from the source"):
        observed(
            tmp_path,
            source(tmp_path),
            single_spectrum=True,
            metadata={"spectra": {"wrong_id": {"cell_id": "cell"}}},
        )


def test_blank_declared_conditions_do_not_support_comparison(tmp_path):
    metadata = comparison_metadata()
    metadata["defaults"]["instrument"] = ""
    with pytest.raises(ValueError, match="must not be blank"):
        observed(tmp_path, comparison_source(tmp_path), metadata=metadata)
