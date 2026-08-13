from __future__ import annotations

import numpy as np
import pandas as pd

from battery_feature_lab.analysis.degradation_tags import (
    build_degradation_tags,
    mann_kendall_sen_slope,
)
from battery_feature_lab.schemas import DiagnosticConfig


def _stress_row(cell_id: str, high_soc_rest: float, c_rate_variance: float) -> dict:
    return {
        "cell_id": cell_id,
        "soc_source": "provided",
        "high_soc_rest_fraction": high_soc_rest,
        "c_rate_variance": c_rate_variance,
        "max_instant_discharge_c_rate": 1.0,
    }


def test_mann_kendall_sen_slope_detects_monotonic_fade() -> None:
    x = np.arange(20)
    y = 1.0 - 0.01 * x
    trend = mann_kendall_sen_slope(x, y)
    assert trend["p_value"] < 0.05
    assert trend["sen_slope"] < 0


def test_single_cell_high_soc_rest_requires_explicit_soc_source() -> None:
    high = pd.DataFrame([_stress_row("A", high_soc_rest=0.6, c_rate_variance=0.01)])
    tags = build_degradation_tags(stress_features=high, config=DiagnosticConfig())
    assert "high_soc_rest_exposure" in set(tags["signal"])

    missing_source = high.drop(columns=["soc_source"])
    tags_without_source = build_degradation_tags(
        stress_features=missing_source,
        config=DiagnosticConfig(),
    )
    signals = (
        set(tags_without_source["signal"])
        if "signal" in tags_without_source.columns
        else set()
    )
    assert "high_soc_rest_exposure" not in signals


def test_single_cell_low_exposure_does_not_false_fire() -> None:
    low = pd.DataFrame([_stress_row("A", high_soc_rest=0.3, c_rate_variance=0.01)])
    tags = build_degradation_tags(stress_features=low, config=DiagnosticConfig())
    signals = set(tags["signal"]) if "signal" in tags.columns else set()
    assert "high_soc_rest_exposure" not in signals
    assert "dynamic_current_variance" not in signals


def test_batch_outlier_is_labelled_relative() -> None:
    vals = [0.009, 0.010, 0.011, 0.010, 0.009, 0.011, 0.010, 0.010, 0.060]
    batch = pd.DataFrame(
        [
            _stress_row(f"C{i}", high_soc_rest=0.1, c_rate_variance=value)
            for i, value in enumerate(vals)
        ]
    )
    tags = build_degradation_tags(stress_features=batch, config=DiagnosticConfig())
    current_variance = tags[tags["signal"] == "dynamic_current_variance"]
    assert len(current_variance) == 1
    assert current_variance["cell_id"].iloc[0] == "C8"
    assert "batch-relative outlier" in current_variance["evidence"].iloc[0]


def test_robust_mad_outlier_not_masked_by_single_extreme() -> None:
    vals = [0.010, 0.012, 0.011, 0.013, 5.0]
    batch = pd.DataFrame(
        [
            _stress_row(f"C{i}", high_soc_rest=0.1, c_rate_variance=value)
            for i, value in enumerate(vals)
        ]
    )
    tags = build_degradation_tags(stress_features=batch, config=DiagnosticConfig())
    current_variance = tags[tags["signal"] == "dynamic_current_variance"]
    assert len(current_variance) == 1
    assert current_variance["cell_id"].iloc[0] == "C4"
    assert "MAD" in current_variance["evidence"].iloc[0]


def test_datasheet_c_rate_flag_requires_spec() -> None:
    fast = pd.DataFrame([_stress_row("A", high_soc_rest=0.1, c_rate_variance=0.01)])
    fast.loc[0, "max_instant_discharge_c_rate"] = 4.5
    no_spec = build_degradation_tags(stress_features=fast, config=DiagnosticConfig())
    with_spec = build_degradation_tags(
        stress_features=fast,
        config=DiagnosticConfig(datasheet_max_discharge_c_rate=3.0),
    )
    no_signals = set(no_spec["signal"]) if "signal" in no_spec.columns else set()
    assert "high_instantaneous_discharge_rate" not in no_signals
    assert "high_instantaneous_discharge_rate" in set(with_spec["signal"])


def test_observation_tags_do_not_assign_lli_or_lam() -> None:
    cycles = pd.DataFrame(
        {
            "cell_id": ["A"] * 20,
            "cycle_index": np.arange(1, 21),
            "discharge_capacity_ah": 1.0 - 0.01 * np.arange(20),
        }
    )
    tags = build_degradation_tags(cycle_features=cycles, config=DiagnosticConfig())
    assert "capacity_decreasing_trend" in set(tags["signal"])
    assert "possible_modes" not in tags.columns
    rendered = tags.to_json()
    assert "LLI" not in rendered
    assert "LAM_PE" not in rendered
    assert "LAM_NE" not in rendered
