"""Tests for short-window operating and electrochemical summaries."""

from __future__ import annotations

import polars as pl
import pytest

from battery_feature_lab.analysis.characterization import (
    analyze_current_step_summary,
    analyze_current_steps,
    analyze_directional_energy,
    analyze_relaxation_summary,
)
from battery_feature_lab.analysis.schema import AnalysisConfig, make_record, metric


def test_directional_energy_uses_balanced_previous_zoh_window() -> None:
    frame = pl.DataFrame(
        {
            "_source_row": [10, 11, 12, 13, 14, 15],
            "test_time_s": [0.0, 1.0, 3.0, 4.0, 6.0, 7.0],
            "current_a": [1.0, 1.0, 0.0, -1.0, -1.0, 0.0],
            "voltage_v": [4.0, 4.0, 3.5, 3.0, 3.0, 3.5],
        }
    )

    record = analyze_directional_energy(
        frame,
        config=AnalysisConfig(),
        cell_id="cell",
        cycle_id=None,
    )

    values = {name: item["value"] for name, item in record["metrics"].items()}
    assert values["charge_throughput"] == pytest.approx(3.0 / 3600.0)
    assert values["discharge_throughput"] == pytest.approx(3.0 / 3600.0)
    assert values["charge_mean_voltage"] == pytest.approx(4.0)
    assert values["discharge_mean_voltage"] == pytest.approx(3.0)
    assert values["directional_mean_voltage_gap"] == pytest.approx(1.0)
    assert values["charge_discharge_throughput_balance"] == pytest.approx(1.0)
    assert values["balanced_window_energy_return_ratio"] == pytest.approx(0.75)
    assert record["applicability"]["status"] == "applicable"


def test_relaxation_summary_normalizes_recovery_direction() -> None:
    records = []
    deltas = [
        (-0.03, "charge"),
        (-0.02, "charge"),
        (-0.01, "charge"),
        (0.02, "discharge"),
        (0.03, "discharge"),
        (0.04, "discharge"),
    ]
    for index, (delta, previous_phase) in enumerate(deltas):
        records.append(
            make_record(
                record_id=f"rest:{index}",
                record_type="response.rest_and_thermal",
                cell_id="cell",
                cycle_scope=None,
                source_intervals=[
                    {
                        "start_row": index * 10,
                        "end_row": index * 10 + 9,
                        "start_time_s": index * 100.0,
                        "end_time_s": index * 100.0 + 60.0,
                    }
                ],
                attributes={"previous_phase": previous_phase},
                metrics={
                    "voltage_change_at_30s": metric(delta, "V"),
                    "voltage_change_at_60s": metric(2 * delta, "V"),
                },
                provider="PyProBE+BFL",
                method_name="test",
                provider_version="test",
            )
        )

    record = analyze_relaxation_summary(
        records, config=AnalysisConfig(), cell_id="cell", cycle_id=None
    )

    metrics = record["metrics"]
    assert metrics["eligible_rest_count_30s"]["value"] == 6
    assert metrics["polarization_recovery_q50_30s"]["value"] == pytest.approx(0.025)
    assert metrics["positive_recovery_fraction_30s"]["value"] == 1.0
    assert len(record["source_intervals"]) == 6
    assert record["applicability"]["status"] == "applicable"


def test_relaxation_signature_requires_phase_conditioned_rests() -> None:
    records = [
        make_record(
            record_id=f"rest:{index}",
            record_type="response.rest_and_thermal",
            cell_id="cell",
            cycle_scope=None,
            source_intervals=[
                {
                    "start_row": index,
                    "end_row": index,
                    "start_time_s": float(index),
                    "end_time_s": float(index + 60),
                }
            ],
            attributes={"previous_phase": None},
            metrics={"voltage_change_at_30s": metric(0.01, "V")},
            provider="PyProBE+BFL",
            method_name="test",
            provider_version="test",
        )
        for index in range(3)
    ]

    record = analyze_relaxation_summary(
        records, config=AnalysisConfig(), cell_id="cell", cycle_id=None
    )

    assert record["metrics"]["eligible_rest_count_30s"]["value"] == 0
    assert record["metrics"]["unconditioned_rest_segment_count"]["value"] == 3
    assert record["applicability"]["status"] == "not_computable"


def test_relaxation_signature_keeps_two_conditioned_members_as_partial() -> None:
    records = []
    for index, (delta, previous_phase) in enumerate([(-0.02, "charge"), (0.7, "discharge")]):
        records.append(
            make_record(
                record_id=f"rest:{index}",
                record_type="response.rest_and_thermal",
                cell_id="cell",
                cycle_scope=None,
                source_intervals=[
                    {
                        "start_row": index,
                        "end_row": index + 1,
                        "start_time_s": float(index * 100),
                        "end_time_s": float(index * 100 + 60),
                    }
                ],
                attributes={"previous_phase": previous_phase, "preceding_mode": "test"},
                metrics={
                    "duration": metric(60.0, "s"),
                    "voltage_change_at_60s": metric(delta, "V"),
                },
                provider="PyProBE+BFL",
                method_name="test",
                provider_version="test",
            )
        )

    record = analyze_relaxation_summary(
        records, config=AnalysisConfig(), cell_id="cell", cycle_id=None
    )

    assert record["applicability"]["status"] == "partial"
    assert record["metrics"]["eligible_rest_count_60s"]["value"] == 2
    assert record["metrics"]["absolute_voltage_change_q50_60s"]["status"] == "not_computable"
    assert [item["record_id"] for item in record["attributes"]["member_summaries"]] == [
        "rest:0",
        "rest:1",
    ]


def test_current_step_extracts_apparent_resistance_without_soc() -> None:
    time = [float(value) for value in range(73)]
    current = [0.0] * 61 + [-2.0] * 12
    voltage = [4.0] * 61 + [3.98] + [3.96] * 11
    frame = pl.DataFrame(
        {
            "_source_row": list(range(73)),
            "test_time_s": time,
            "current_a": current,
            "voltage_v": voltage,
        }
    )
    phases = [
        {
            "segment_index": 0,
            "start": 0,
            "end": 61,
            "phase": "rest",
            "duration_s": 61.0,
            "source_interval": {
                "start_row": 0,
                "end_row": 60,
                "start_time_s": 0.0,
                "end_time_s": 61.0,
            },
        },
        {
            "segment_index": 1,
            "start": 61,
            "end": 73,
            "phase": "discharge",
            "duration_s": 12.0,
            "next_phase": None,
            "source_interval": {
                "start_row": 61,
                "end_row": 72,
                "start_time_s": 61.0,
                "end_time_s": 73.0,
            },
        },
    ]

    records = analyze_current_steps(
        frame,
        phases,
        response_times_s=(2.0, 10.0),
        config=AnalysisConfig(),
        cell_id="cell",
        cycle_id=None,
    )
    summary = analyze_current_step_summary(
        records,
        response_times_s=(2.0, 10.0),
        config=AnalysisConfig(),
        cell_id="cell",
        cycle_id=None,
    )

    record = records[0]
    assert record["attributes"]["direction"] == "discharge"
    assert record["attributes"]["reference_frame"]["soc_reference"] is None
    assert record["metrics"]["apparent_dc_resistance_first_valid"]["value"] == pytest.approx(0.01)
    assert record["metrics"]["apparent_dc_resistance_10s"]["value"] == pytest.approx(0.02)
    assert record["metrics"]["checkpoint_bracket_width_10s"]["value"] == 0.0
    assert record["metrics"]["pre_step_temperature"]["status"] == "not_computable"
    assert {item["role"] for item in record["source_intervals"]} == {
        "pre_step_baseline",
        "current_step_response",
    }
    assert summary["metrics"]["computed_step_count"]["value"] == 1
    assert summary["metrics"]["candidate_step_count"]["value"] == 1
    assert summary["metrics"]["rejected_step_count"]["value"] == 0
    assert summary["applicability"]["status"] == "not_computable"


def test_current_step_summary_audits_rejections_without_generic_child_flag() -> None:
    rejected = make_record(
        record_id="response.current_step:none:1",
        record_type="response.current_step",
        cell_id="cell",
        cycle_scope=None,
        source_intervals=[{"start_row": 0, "end_row": 10}],
        attributes={"candidate_detected": True},
        provider="BFL",
        method_name="current_step_delta_v_over_delta_i_v1",
        provider_version="0.4.0",
        applicability_status="not_computable",
        applicability_reasons=["sampling_interval_too_sparse_for_current_step"],
        quality_status="warning",
        quality_flags=["input_not_eligible"],
    )

    summary = analyze_current_step_summary(
        [rejected],
        response_times_s=(2.0, 10.0),
        config=AnalysisConfig(),
        cell_id="cell",
        cycle_id=None,
    )

    assert summary["metrics"]["candidate_step_count"]["value"] == 1
    assert summary["metrics"]["rejected_step_count"]["value"] == 1
    assert summary["attributes"]["rejection_counts_by_reason"] == {
        "sampling_interval_too_sparse_for_current_step": 1
    }
    assert summary["quality"]["flags"] == ["current_step_candidates_rejected"]
