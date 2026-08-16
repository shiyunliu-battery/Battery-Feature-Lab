"""Tests for capacity reference and robust evolution statistics."""

from __future__ import annotations

from pathlib import Path

import pytest

from battery_feature_lab.analysis.evolution import analyze_capacity_evolution
from battery_feature_lab.analysis.schema import AnalysisConfig


def test_capacity_evolution_uses_reference_median_theil_sen_and_tau_b() -> None:
    summaries = []
    for cycle in range(1, 11):
        summaries.append(
            {
                "cycle_id": cycle,
                "cycle_id_source": "source",
                "complete": True,
                "discharge_capacity_ah": 1.0 - 0.01 * cycle,
                "operation_signature": "R-C-R-D-R",
                "source_interval": {"start_row": cycle * 10, "end_row": cycle * 10 + 9},
            }
        )
    config = AnalysisConfig(output_dir=Path("unused"))

    record = analyze_capacity_evolution(summaries, config=config, cell_id="cell", provider_calls=[])

    assert record["attributes"]["reference_cycle_ids"] == [2, 3, 4, 5]
    assert record["metrics"]["reference_discharge_capacity"]["value"] == pytest.approx(0.965)
    assert record["metrics"]["capacity_retention"]["value"] == pytest.approx(0.9 / 0.965)
    assert record["metrics"]["theil_sen_slope_per_100_cycles"]["value"] == pytest.approx(-1.0)
    assert record["metrics"]["kendall_tau_b"]["value"] == pytest.approx(-1.0)


def test_inferred_cycle_ids_are_gated_out() -> None:
    record = analyze_capacity_evolution(
        [
            {
                "cycle_id": 1,
                "cycle_id_source": "inferred",
                "complete": True,
                "discharge_capacity_ah": 1.0,
                "operation_signature": "C-D",
                "source_interval": {},
            }
        ],
        config=AnalysisConfig(),
        cell_id="cell",
        provider_calls=[],
    )
    assert record["applicability"]["status"] == "not_computable"
    assert record["metrics"]["capacity_retention"]["value"] is None


def test_early_cycle_exclusion_precedes_completeness_filter() -> None:
    summaries = [
        {
            "cycle_id": 1,
            "cycle_id_source": "source",
            "complete": False,
            "discharge_capacity_ah": None,
            "operation_signature": "C-D",
        },
        *[
            {
                "cycle_id": cycle,
                "cycle_id_source": "source",
                "complete": True,
                "discharge_capacity_ah": 1.0,
                "operation_signature": "C-D",
                "source_interval": None,
            }
            for cycle in range(2, 5)
        ],
    ]

    record = analyze_capacity_evolution(
        summaries,
        config=AnalysisConfig(formation_cycles_to_exclude=1),
        cell_id="cell",
        provider_calls=[],
    )

    assert record["attributes"]["excluded_cycle_ids"] == [1]
    assert record["attributes"]["comparable_cycle_ids"] == [2, 3, 4]
