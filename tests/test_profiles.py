"""Capacity-aligned profile eligibility and no-extrapolation tests."""

from __future__ import annotations

import numpy as np
import polars as pl

from battery_feature_lab.analysis.operation import _phase_segments
from battery_feature_lab.analysis.profiles import analyze_capacity_aligned_profile
from battery_feature_lab.analysis.schema import AnalysisConfig


def _paired_frame(*, charge_start_v: float = 2.5) -> pl.DataFrame:
    lengths = [61, 1001, 61, 1001, 61]
    current = np.concatenate(
        [
            np.zeros(lengths[0]),
            np.ones(lengths[1]),
            np.zeros(lengths[2]),
            -np.ones(lengths[3]),
            np.zeros(lengths[4]),
        ]
    )
    voltage = np.concatenate(
        [
            np.full(lengths[0], charge_start_v),
            np.linspace(charge_start_v, 4.2, lengths[1]),
            np.full(lengths[2], 4.2),
            np.linspace(4.2, 2.5, lengths[3]),
            np.full(lengths[4], 2.5),
        ]
    )
    steps = np.repeat(np.arange(5), lengths)
    return pl.DataFrame(
        {
            "_source_row": np.arange(len(current)),
            "test_time_s": np.arange(len(current), dtype=float),
            "current_a": current,
            "voltage_v": voltage,
            "step_index": steps,
        }
    )


def _nonadjacent_match_frame() -> pl.DataFrame:
    block_length = 1001
    rest_length = 61
    current = np.concatenate(
        [
            np.zeros(rest_length),
            np.ones(block_length),
            np.zeros(rest_length),
            -0.5 * np.ones(block_length),
            np.zeros(rest_length),
            0.2 * np.ones(block_length),
            np.zeros(rest_length),
            -np.ones(block_length),
            np.zeros(rest_length),
        ]
    )
    voltage = np.concatenate(
        [
            np.full(rest_length, 2.5),
            np.linspace(2.5, 4.2, block_length),
            np.full(rest_length, 4.2),
            np.linspace(4.2, 3.35, block_length),
            np.full(rest_length, 3.35),
            np.linspace(3.35, 3.6, block_length),
            np.full(rest_length, 3.6),
            np.linspace(4.2, 2.5, block_length),
            np.full(rest_length, 2.5),
        ]
    )
    lengths = [61, 1001, 61, 1001, 61, 1001, 61, 1001, 61]
    steps = np.concatenate([np.full(length, index) for index, length in enumerate(lengths)])
    return pl.DataFrame(
        {
            "_source_row": np.arange(len(current)),
            "test_time_s": np.arange(len(current), dtype=float),
            "current_a": current,
            "voltage_v": voltage,
            "step_index": steps,
        }
    )


def test_capacity_aligned_profile_emits_101_point_shared_upper_axis() -> None:
    frame = _paired_frame()
    phases = _phase_segments(frame, 1e-4)

    record = analyze_capacity_aligned_profile(
        frame,
        phases,
        config=AnalysisConfig(),
        cell_id="cell",
        cycle_id=None,
        cycle_id_source="absent",
    )

    assert record["applicability"]["status"] == "applicable"
    assert len(record["series"]["capacity_from_shared_upper_endpoint_ah"]) == 101
    assert record["metrics"]["directional_capacity_balance"]["value"] == 1.0
    assert record["attributes"]["reference_frame"]["is_soc"] is False
    assert record["series"]["charge_current_a"][0] == 1.0
    assert record["series"]["charge_current_a"][-1] == 1.0
    assert record["series"]["discharge_current_a"][0] == 1.0
    assert record["series"]["discharge_current_a"][-1] == 1.0
    assert {item["role"] for item in record["source_intervals"]} == {
        "leading_rest",
        "charge_member",
        "trailing_rest",
        "discharge_member",
    }


def test_capacity_aligned_profile_rejects_partial_charge_state_window() -> None:
    frame = _paired_frame(charge_start_v=3.5)
    phases = _phase_segments(frame, 1e-4)

    record = analyze_capacity_aligned_profile(
        frame,
        phases,
        config=AnalysisConfig(),
        cell_id="cell",
        cycle_id=None,
        cycle_id_source="absent",
    )

    assert record["applicability"]["status"] == "not_computable"
    assert (
        "charge_voltage_state_span_does_not_cover_discharge_span"
        in record["applicability"]["reasons"]
    )
    assert record["series"] == {}


def test_capacity_profile_does_not_pair_nonadjacent_full_blocks() -> None:
    frame = _nonadjacent_match_frame()
    phases = _phase_segments(frame, 1e-4)

    record = analyze_capacity_aligned_profile(
        frame,
        phases,
        config=AnalysisConfig(),
        cell_id="cell",
        cycle_id=None,
        cycle_id_source="absent",
    )

    assert record["applicability"]["status"] == "not_computable"
    assert record["series"] == {}
    assert record["metrics"]["eligible_pair_count"]["value"] == 0


def test_capacity_profile_policy_override_changes_rest_eligibility() -> None:
    frame = _paired_frame()
    phases = _phase_segments(frame, 1e-4)

    record = analyze_capacity_aligned_profile(
        frame,
        phases,
        config=AnalysisConfig(
            analysis_policy={"profile_bracketing_rest_min_s": 61.0}
        ),
        cell_id="cell",
        cycle_id=None,
        cycle_id_source="absent",
    )

    assert record["applicability"]["status"] == "not_computable"
    assert any(
        "directional_block_not_bracketed_by_61s_rests"
        in candidate["eligibility_reasons"]
        for candidate in record["attributes"]["candidate_trajectories"]
    )
    assert (
        record["method"]["parameters"]["minimum_bracketing_rest_duration_s"]
        == 61.0
    )
