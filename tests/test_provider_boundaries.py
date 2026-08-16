"""Third-party failures remain explicit analysis records instead of aborting."""

from __future__ import annotations

from battery_feature_lab.analysis import evolution
from battery_feature_lab.analysis.schema import AnalysisConfig


def test_theil_sen_provider_error_does_not_abort_evolution(monkeypatch) -> None:
    summaries = [
        {
            "cycle_id": cycle,
            "cycle_id_source": "source",
            "complete": True,
            "discharge_capacity_ah": 1.0 - cycle * 0.001,
            "operation_signature": "C-R-D",
            "source_interval": {
                "start_row": cycle * 10,
                "end_row": cycle * 10 + 9,
                "start_time_s": cycle * 100.0,
                "end_time_s": cycle * 100.0 + 99.0,
            },
        }
        for cycle in range(1, 9)
    ]

    def fail(*args, **kwargs):
        raise RuntimeError("theil failure")

    monkeypatch.setattr(evolution, "theilslopes", fail)
    provider_calls = []
    record = evolution.analyze_capacity_evolution(
        summaries,
        config=AnalysisConfig(formation_cycles_to_exclude=0),
        cell_id="cell",
        provider_calls=provider_calls,
    )

    slope = record["metrics"]["theil_sen_slope_per_100_cycles"]
    assert slope["status"] == "not_computable"
    assert slope["reason"] == "provider_error: theil failure"
    assert any(
        call["method"] == "stats.theilslopes" and call["status"] == "error"
        for call in provider_calls
    )
