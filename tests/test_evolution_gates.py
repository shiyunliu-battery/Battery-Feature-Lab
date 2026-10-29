"""Capacity evolution does not treat different discharge rates as repeat measurements."""

from __future__ import annotations

import pytest

from battery_feature_lab.analysis.evolution import analyze_capacity_evolution
from battery_feature_lab.analysis.schema import AnalysisConfig


def summary(cycle, current, capacity=1.0, temperature=None, signature="R-C-R-D-R"):
    return {
        "cycle_id": cycle,
        "cycle_id_source": "source",
        "complete": True,
        "discharge_capacity_ah": capacity,
        "operation_signature": signature,
        "discharge_current_a": current,
        "discharge_temperature_deg_c": temperature,
        "source_interval": {"start_row": cycle * 10, "end_row": cycle * 10 + 9},
    }


def evolve(summaries, **policy):
    return analyze_capacity_evolution(
        summaries,
        config=AnalysisConfig(formation_cycles_to_exclude=0, analysis_policy=policy),
        cell_id="cell",
        provider_calls=[],
    )


def test_one_cycle_per_rate_has_no_repeat_measurement():
    # Chen et al. (2020) layout: one discharge each at C/10, C/2, 1C and 1.5C of a 5 Ah cell.
    record = evolve(
        [
            summary(1, 0.5, 4.998),
            summary(2, 2.5, 4.924),
            summary(3, 5.0, 4.85),
            summary(4, 7.5, 4.8),
        ]
    )
    assert record["applicability"]["status"] == "not_computable"
    assert record["applicability"]["reasons"] == ["no_two_cycles_share_operation_conditions"]
    assert record["attributes"]["comparable_cycle_ids"] == []
    assert record["metrics"]["comparable_capacity_range"]["value"] is None


def test_rate_check_ups_are_separated_from_ageing_cycles():
    cycles = [summary(c, 5.0, 1.0 - 0.001 * c) for c in range(1, 11)]
    cycles += [summary(c, 0.25, 1.05) for c in (11, 12)]
    record = evolve(cycles)
    assert record["attributes"]["comparable_cycle_ids"] == list(range(1, 11))
    assert record["metrics"]["comparable_discharge_current_median"]["value"] == pytest.approx(5.0)
    assert record["method"]["parameters"]["discharge_current_relative_tolerance"] == 0.05


def test_small_current_differences_stay_within_one_class():
    record = evolve([summary(c, 5.0 + 0.01 * c) for c in range(1, 6)])
    assert record["attributes"]["comparable_cycle_ids"] == [1, 2, 3, 4, 5]
    relative_range = record["metrics"]["comparable_discharge_current_relative_range"]["value"]
    assert relative_range == pytest.approx(0.04 / 5.03)


def test_tied_repeated_groups_remain_ambiguous():
    record = evolve([summary(c, 5.0) for c in (1, 2)] + [summary(c, 2.5) for c in (3, 4)])
    assert record["applicability"]["reasons"] == ["ambiguous_dominant_operation_signature"]


def test_missing_discharge_current_is_not_evidence_of_comparability():
    record = evolve([summary(c, None) for c in range(1, 4)])
    assert record["applicability"]["reasons"] == ["missing_discharge_current"]


def test_temperature_spread_above_policy_is_flagged():
    record = evolve([summary(c, 5.0, temperature=25.0 + c) for c in range(1, 5)])
    assert record["metrics"]["comparable_discharge_temperature_range"]["value"] == pytest.approx(
        3.0
    )
    assert "comparable_cycle_temperature_difference_exceeds_policy" in record["quality"]["flags"]
    assert record["quality"]["status"] == "warning"

    relaxed = evolve(
        [summary(c, 5.0, temperature=25.0 + c) for c in range(1, 5)],
        evolution_temperature_difference_max_deg_c=5.0,
    )
    assert relaxed["quality"]["flags"] == []


def test_unavailable_temperature_is_reported():
    record = evolve([summary(c, 5.0) for c in range(1, 4)])
    temperature = record["metrics"]["comparable_discharge_temperature_range"]
    assert temperature["status"] == "not_computable"
    assert temperature["reason"] == "discharge_temperature_unavailable"
