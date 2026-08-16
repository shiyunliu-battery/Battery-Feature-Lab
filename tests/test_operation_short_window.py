"""Short-window operation segmentation and exposure semantics."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from battery_feature_lab.analysis.operation import (
    _exposure_record,
    _phase_segments,
    analyze_operation,
)
from battery_feature_lab.analysis.schema import AnalysisConfig


def test_phase_segments_respect_source_steps_and_do_not_call_nan_discharge() -> None:
    frame = pl.DataFrame(
        {
            "_source_row": [0, 1, 2, 3],
            "record_index": [10, 11, 12, 13],
            "test_time_s": [0.0, 1.0, 2.0, 3.0],
            "current_a": [1.0, 1.0, 1.0, np.nan],
            "step_index": [0, 0, 1, 1],
        }
    )

    segments = _phase_segments(frame, threshold=1e-4)

    assert [item["phase"] for item in segments] == [
        "charge",
        "charge",
        "unknown",
    ]
    assert [item["source_step_index"] for item in segments] == [0, 1, 1]
    assert segments[-1]["flags"] == [
        "fewer_than_three_samples",
        "non_finite_current_phase",
    ]


def test_c_rate_uses_absolute_current_not_signed_cancellation() -> None:
    frame = pl.DataFrame(
        {
            "_source_row": [0, 1, 2],
            "test_time_s": [0.0, 1.0, 2.0],
            "current_a": [2.0, -2.0, 0.0],
        }
    )

    record = _exposure_record(
        frame,
        AnalysisConfig(nominal_capacity_ah=2.0),
        "cell",
        None,
    )

    assert record["metrics"]["mean"]["value"] == 0.0
    assert record["metrics"]["absolute_current_mean"]["value"] == 2.0
    assert record["metrics"]["c_rate_mean"]["value"] == 1.0


def test_current_only_operation_uses_current_shape_without_pyprobe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pl.DataFrame(
        {
            "_source_row": list(range(18)),
            "record_index": list(range(100, 118)),
            "test_time_s": [
                0.0,
                30.0,
                60.0,
                61.0,
                66.0,
                70.0,
                71.0,
                101.0,
                131.0,
                132.0,
                152.0,
                172.0,
                192.0,
                193.0,
                194.0,
                195.0,
                196.0,
                197.0,
            ],
            "current_a": [
                0.0,
                0.0,
                0.0,
                2.0,
                2.0,
                2.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -0.2,
                -1.0,
                -2.0,
                -0.3,
                -0.8,
            ],
            "step_index": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        }
    )
    provider_calls: list[dict[str, object]] = []

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("PyProBE must not be constructed for a current-only frame")

    monkeypatch.setattr(
        "battery_feature_lab.analysis.operation.pyprobe_experiment",
        fail_if_called,
    )

    records, phases = analyze_operation(
        frame,
        None,
        config=AnalysisConfig(),
        cell_id="current-only",
        cycle_id=None,
        provider_calls=provider_calls,
    )

    assert phases
    assert {record["record_type"] for record in records} >= {
        "operation.phase_segment",
        "operation.mode_segment",
        "operation.window_summary",
        "operation.exposure_summary",
    }
    mode_records = [
        record for record in records if record["record_type"] == "operation.mode_segment"
    ]
    modes = {record["attributes"]["mode"] for record in mode_records}
    assert {"pulse_like", "constant_current_like", "dynamic_current"} <= modes
    assert "constant_voltage_like" not in modes
    assert all(record["method"]["provider"] == "BFL" for record in mode_records)
    active_mode_records = [
        record
        for record in mode_records
        if record["attributes"]["phase"] in {"charge", "discharge"}
    ]
    assert all(
        record["attributes"]["mode_provider_branch"] == "bfl_current_shape_only"
        for record in active_mode_records
    )
    assert all(
        record["attributes"]["pyprobe_candidate_rows"]
        == {"constant_current": None, "constant_voltage": None}
        for record in active_mode_records
    )

    window = next(
        record for record in records if record["record_type"] == "operation.window_summary"
    )
    assert window["metrics"]["capacity_throughput"]["status"] == "ok"
    assert window["metrics"]["current_squared_exposure"]["status"] == "ok"
    for name in (
        "energy_throughput",
        "absolute_power_q95",
        "voltage_min",
        "voltage_q05",
        "voltage_q50",
        "voltage_q95",
        "voltage_max",
    ):
        assert window["metrics"][name] == {
            "value": None,
            "unit": window["metrics"][name]["unit"],
            "status": "not_computable",
            "reason": "missing_required_channel:voltage_v",
        }
    for name in (
        "temperature_min",
        "temperature_q50",
        "temperature_q95",
        "temperature_max",
        "temperature_duration_coverage",
    ):
        assert window["metrics"][name]["status"] == "not_computable"
        assert window["metrics"][name]["reason"] == "missing_required_channel:temperature"

    assert provider_calls
    assert all(call["status"] == "not_invoked" for call in provider_calls)
    assert all(call["reason"] == "missing_required_channel:voltage_v" for call in provider_calls)


def test_window_absolute_current_preserves_symmetric_charge_and_discharge() -> None:
    frame = pl.DataFrame(
        {
            "_source_row": [0, 1, 2],
            "test_time_s": [0.0, 1.0, 2.0],
            "current_a": [2.0, -2.0, 0.0],
        }
    )

    records, _ = analyze_operation(
        frame,
        None,
        config=AnalysisConfig(),
        cell_id="symmetric-current",
        cycle_id=None,
        provider_calls=[],
    )

    window = next(
        record for record in records if record["record_type"] == "operation.window_summary"
    )
    exposure = next(
        record for record in records if record["record_type"] == "operation.exposure_summary"
    )
    assert exposure["metrics"]["mean"]["value"] == 0.0
    assert window["metrics"]["absolute_current_q50"]["value"] == 2.0
    assert window["metrics"]["absolute_current_q95"]["value"] == 2.0
    assert window["metrics"]["capacity_throughput"]["value"] == pytest.approx(4.0 / 3600.0)
    assert window["metrics"]["current_squared_exposure"]["value"] == pytest.approx(8.0 / 3600.0)


def test_operation_policy_override_changes_current_stability_gate() -> None:
    frame = pl.DataFrame(
        {
            "_source_row": [0, 1, 2, 3],
            "test_time_s": [0.0, 30.0, 60.0, 90.0],
            "current_a": [1.0, 1.2, 0.8, 1.0],
        }
    )

    default_records, _ = analyze_operation(
        frame,
        None,
        config=AnalysisConfig(),
        cell_id="policy-default",
        cycle_id=None,
        provider_calls=[],
    )
    override_records, _ = analyze_operation(
        frame,
        None,
        config=AnalysisConfig(
            analysis_policy={
                "mode_cc_current_cv_max": 0.1,
                "mode_taper_absolute_noise_a": 0.25,
                "mode_taper_relative_noise_fraction": 0.5,
            }
        ),
        cell_id="policy-override",
        cycle_id=None,
        provider_calls=[],
    )
    minimum_records, _ = analyze_operation(
        frame,
        None,
        config=AnalysisConfig(analysis_policy={"mode_min_samples": 5}),
        cell_id="policy-minimum-samples",
        cycle_id=None,
        provider_calls=[],
    )

    default_mode = next(
        record for record in default_records if record["record_type"] == "operation.mode_segment"
    )
    override_mode = next(
        record for record in override_records if record["record_type"] == "operation.mode_segment"
    )
    override_window = next(
        record for record in override_records if record["record_type"] == "operation.window_summary"
    )
    minimum_mode = next(
        record for record in minimum_records if record["record_type"] == "operation.mode_segment"
    )
    assert default_mode["attributes"]["mode"] == "constant_current_like"
    assert override_mode["attributes"]["mode"] == "dynamic_current"
    assert minimum_mode["attributes"]["mode"] == "unmatched"
    assert "fewer_than_mode_minimum_samples" in minimum_mode["quality"]["flags"]
    parameters = override_mode["method"]["parameters"]
    assert parameters["cc_current_cv_limit"] == 0.1
    assert parameters["minimum_sample_count"] == 3
    assert parameters["cv_taper_noise_absolute_floor_a"] == 0.25
    assert parameters["cv_taper_noise_relative_factor"] == 0.5
    assert parameters["dominant_classified_fraction_min"] == 0.5
    assert override_mode["attributes"]["taper_noise_tolerance_a"] == pytest.approx(0.6)
    assert override_window["method"]["parameters"]["mode_policy"]["cc_current_cv_limit"] == 0.1


def test_dominant_mode_fraction_policy_changes_window_gate() -> None:
    frame = pl.DataFrame(
        {
            "_source_row": [0, 1, 2, 3, 4, 5],
            "test_time_s": [0.0, 30.0, 59.0, 60.0, 100.0, 101.0],
            "current_a": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "step_index": [0, 0, 0, 1, 2, 2],
        }
    )

    default_records, _ = analyze_operation(
        frame,
        None,
        config=AnalysisConfig(),
        cell_id="dominant-default",
        cycle_id=None,
        provider_calls=[],
    )
    strict_records, _ = analyze_operation(
        frame,
        None,
        config=AnalysisConfig(analysis_policy={"mode_dominant_classified_fraction_min": 0.8}),
        cell_id="dominant-strict",
        cycle_id=None,
        provider_calls=[],
    )

    default_window = next(
        record for record in default_records if record["record_type"] == "operation.window_summary"
    )
    strict_window = next(
        record for record in strict_records if record["record_type"] == "operation.window_summary"
    )
    assert default_window["attributes"]["dominant_active_mode"] == "constant_current_like"
    assert strict_window["attributes"]["dominant_active_mode"] is None
    assert strict_window["attributes"]["dominant_mode_minimum_classified_active_fraction"] == 0.8
