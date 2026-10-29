"""Lin-KK series-capacitance default and EIS comparison gates."""

from __future__ import annotations

import importlib.util

import numpy as np
import polars as pl
import pytest

import bfl
from battery_feature_lab.eis.schema import EISConfig

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("eisfit", "impedance")),
    reason="EIS regression tests require battery-feature-lab[eis]",
)


def find(result, kind):
    return next(r for r in result.records if r["record_type"] == kind)


def diffusion_tail_source(tmp_path):
    frequency = np.logspace(4, -2, 40)
    omega = 2 * np.pi * frequency
    impedance = 0.01 + 0.02 / (1 + 1j * omega * 0.05) + 0.004 * (1 - 1j) / np.sqrt(omega)
    path = tmp_path / "tail.csv"
    pl.DataFrame(
        {"Frequency_Hz": frequency, "Real_Ohm": impedance.real, "Imag_Ohm": impedance.imag}
    ).write_csv(path)
    return path


def test_lin_kk_uses_a_series_capacitance_by_default(tmp_path):
    result = bfl.analyze_eis(
        diffusion_tail_source(tmp_path), tmp_path / "out", single_spectrum=True, run_fit=False
    )
    kk = find(result, "response.eis_kk")
    assert kk["method"]["parameters"]["add_cap"] is True
    assert kk["attributes"]["measurement_consistency"] == "within_configured_residual_limit"


def test_lin_kk_without_series_capacitance_flags_the_diffusion_tail(tmp_path):
    result = bfl.analyze_eis(
        diffusion_tail_source(tmp_path),
        tmp_path / "out",
        single_spectrum=True,
        run_fit=False,
        kk_options={"add_cap": False},
    )
    kk = find(result, "response.eis_kk")
    assert kk["method"]["parameters"]["add_cap"] is False
    assert kk["attributes"]["measurement_consistency"] == "outside_configured_residual_limit"
    assert kk["metrics"]["frequency_of_maximum_residual"]["value"] == pytest.approx(0.01)


def test_lin_kk_add_cap_must_be_boolean():
    with pytest.raises(TypeError, match="add_cap"):
        EISConfig(kk_options={"add_cap": "yes"})


def two_spectra(tmp_path, *, columns_a=None, columns_b=None):
    frequency = np.logspace(4, -1, 32)
    impedance = 0.01 + 0.02 / (1 + 2j * np.pi * frequency * 0.1)
    frames = []
    for name, extra in (("a", columns_a or {}), ("b", columns_b or {})):
        frames.append(
            pl.DataFrame(
                {
                    "Spectrum_ID": [name] * len(frequency),
                    "Frequency_Hz": frequency,
                    "Real_Ohm": impedance.real,
                    "Imag_Ohm": impedance.imag,
                    **{key: [value] * len(frequency) for key, value in extra.items()},
                }
            )
        )
    path = tmp_path / "pair.csv"
    pl.concat(frames, how="diagonal_relaxed").write_csv(path)
    return path


def defaults(**overrides):
    values = {
        "cell_id": "cell",
        "temperature_deg_c": 25.0,
        "soc_fraction": 0.5,
        "rest_time_s": 3600.0,
        "instrument": "instrument",
        "protocol_id": "protocol",
        "ac_amplitude_v": 0.01,
    }
    values.update(overrides)
    return values


def compare(tmp_path, metadata, kind="repeat", **options):
    result = bfl.analyze_eis(
        two_spectra(tmp_path, **options.pop("columns", {})),
        tmp_path / "out",
        run_kk=False,
        run_fit=False,
        metadata=metadata,
        comparison_pairs=(("a", "b"),),
        comparison_kind=kind,
        **options,
    )
    return find(result, "evolution.eis_comparison")


def test_dc_current_difference_makes_spectra_incompatible(tmp_path):
    record = compare(
        tmp_path,
        {
            "defaults": defaults(),
            "spectra": {"a": {"dc_current_a": 0.0}, "b": {"dc_current_a": 0.5}},
        },
    )
    assert record["attributes"]["comparison_status"] == "incompatible"
    assert "dc_current_a" in record["attributes"]["mismatched_conditions"]
    assert record["attributes"]["unassessed_conditions"] == []


def test_dc_current_is_read_from_the_source_column(tmp_path):
    record = compare(
        tmp_path,
        {"defaults": defaults()},
        columns={"columns_a": {"DC_Current_A": 0.0}, "columns_b": {"DC_Current_A": -0.5}},
    )
    assert "dc_current_a" in record["attributes"]["mismatched_conditions"]


def test_dc_current_recorded_for_one_spectrum_only_is_missing(tmp_path):
    record = compare(tmp_path, {"defaults": defaults(), "spectra": {"a": {"dc_current_a": 0.0}}})
    assert record["attributes"]["comparison_status"] == "conditional"
    assert "dc_current_a" in record["attributes"]["missing_conditions"]


def test_unrecorded_dc_current_is_listed_as_unassessed(tmp_path):
    record = compare(tmp_path, {"defaults": defaults()})
    assert record["attributes"]["comparison_status"] == "comparable"
    assert record["attributes"]["unassessed_conditions"] == ["dc_current_a"]


@pytest.mark.parametrize("temperature_b,expected", [(25.02, "comparable"), (26.0, "incompatible")])
def test_measured_temperature_uses_a_default_tolerance(tmp_path, temperature_b, expected):
    record = compare(
        tmp_path,
        {"defaults": defaults(), "spectra": {"b": {"temperature_deg_c": temperature_b}}},
    )
    assert record["attributes"]["comparison_status"] == expected
    assert record["method"]["parameters"]["tolerances"]["temperature_deg_c"] == 0.5
    assert record["method"]["parameters"]["tolerance_sources"]["temperature_deg_c"] == "default"


def test_declared_tolerance_overrides_the_default(tmp_path):
    record = compare(
        tmp_path,
        {"defaults": defaults(), "spectra": {"b": {"temperature_deg_c": 25.02}}},
        comparison_tolerances={"temperature_deg_c": 0.0},
    )
    assert record["attributes"]["comparison_status"] == "incompatible"
    assert record["method"]["parameters"]["tolerance_sources"]["temperature_deg_c"] == "declared"


def test_specimen_comparison_lets_only_the_cell_differ(tmp_path):
    metadata = {"defaults": defaults(), "spectra": {"b": {"cell_id": "other-cell"}}}
    specimen = compare(tmp_path, metadata, kind="specimen")
    assert specimen["attributes"]["comparison_status"] == "comparable"
    assert any("between-specimen" in limit for limit in specimen["interpretation_limits"])

    (tmp_path / "repeat").mkdir()
    repeat = compare(tmp_path / "repeat", metadata)
    assert repeat["attributes"]["comparison_status"] == "incompatible"
    assert "cell_id" in repeat["attributes"]["mismatched_conditions"]


def test_specimen_comparison_of_one_cell_is_rejected(tmp_path):
    record = compare(tmp_path, {"defaults": defaults()}, kind="specimen")
    assert "comparison_axis_unchanged" in record["attributes"]["mismatched_conditions"]
