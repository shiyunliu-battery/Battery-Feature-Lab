"""Capacity–voltage curves per phase segment (response.capacity_voltage_curve).

The stored polyline must stay within its stated tolerance of the samples' linear interpolation everywhere,
use the same capacity as the segment's trapezoid integral, keep the knee of a discharge, and say when the
point limit forced a wider tolerance.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import bfl
from battery_feature_lab.analysis import operation
from battery_feature_lab.analysis.curves import (
    analyze_capacity_voltage_curves,
    capacity_voltage_samples,
    simplify_polyline,
)
from battery_feature_lab.analysis.schema import AnalysisConfig

DT = 10.0


def _discharge_with_knee(samples: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(samples) * DT
    current = np.full(samples, -2.5)
    fraction = np.linspace(0.0, 1.0, samples)
    voltage = 4.1 - 0.6 * fraction - 0.0004 * np.exp(fraction * 8.0)  # slope, then a knee near the end
    return time, current, voltage


def _frame() -> pl.DataFrame:
    t, i, v = _discharge_with_knee()
    n = len(t)
    rest_t = t[-1] + DT * np.arange(1, 31)
    charge_t = rest_t[-1] + DT * np.arange(1, 61)
    time = np.r_[t, rest_t, charge_t]
    current = np.r_[i, np.zeros(30), np.full(60, 1.5)]
    voltage = np.r_[v, np.full(30, 3.1), np.linspace(3.2, 4.0, 60)]
    m = len(time)
    return pl.DataFrame({"_source_row": np.arange(m), "record_index": np.arange(1, m + 1), "test_time_s": time,
                         "current_a": current, "voltage_v": voltage,
                         "step_index": np.r_[np.ones(n, dtype=int), np.full(30, 2), np.full(60, 3)]})


def _curves(frame: pl.DataFrame, **policy) -> tuple[list[dict], list[dict]]:
    config = AnalysisConfig(analysis_policy=policy)
    records, phases = operation.analyze_operation(frame, None, config=config, cell_id="cell", cycle_id=None, provider_calls=[])
    curves = analyze_capacity_voltage_curves(frame, phases, config=config, cell_id="cell", cycle_id=None)
    return curves, [r for r in records if r["record_type"] == "operation.phase_segment"]


def test_simplification_keeps_a_line_as_two_points_and_respects_the_bound() -> None:
    x = np.linspace(0.0, 5.0, 200)
    assert simplify_polyline(x, 3.0 - 0.1 * x, 1e-4).tolist() == [0, 199]
    y = np.sin(x)
    for tolerance in (1e-2, 1e-3, 1e-4):
        kept = simplify_polyline(x, y, tolerance)
        assert np.max(np.abs(np.interp(x, x[kept], y[kept]) - y)) <= tolerance
        assert kept[0] == 0 and kept[-1] == len(x) - 1


def test_one_sample_per_capacity_value_the_last() -> None:
    time = np.array([0.0, 10.0, 10.0, 20.0])  # a repeated time stamp adds no capacity
    q, v, total = capacity_voltage_samples(time, np.full(4, -1.0), np.array([3.9, 3.8, 3.7, 3.6]))
    assert q.tolist() == pytest.approx([0.0, 10 / 3600, 20 / 3600])
    assert v.tolist() == [3.9, 3.7, 3.6] and total == pytest.approx(20 / 3600)


def test_discharge_and_charge_segments_get_curves_within_tolerance() -> None:
    frame = _frame()
    curves, segments = _curves(frame)
    assert [c["attributes"]["phase"] for c in curves] == ["discharge", "charge"]
    discharge = curves[0]
    m = discharge["metrics"]
    assert m["max_interpolation_error"]["value"] <= m["voltage_tolerance"]["value"] == 0.0005
    assert 2 < m["point_count"]["value"] < m["distinct_capacity_samples"]["value"] == 400
    capacity = discharge["series"]["capacity_ah"]
    assert capacity[0] == 0.0 and capacity[-1] == pytest.approx(m["capacity"]["value"])
    segment = next(s for s in segments if s["record_id"] == discharge["attributes"]["phase_segment_record_id"])
    assert m["capacity"]["value"] == pytest.approx(segment["metrics"]["trapezoid_capacity"]["value"])
    # the knee is kept: points are denser over the last tenth of the capacity than over the first
    q = np.array(capacity)
    assert np.sum(q > 0.9 * q[-1]) > np.sum(q < 0.1 * q[-1])
    assert discharge["method"]["name"] == "trapezoid_capacity_voltage_polyline_v1"


def test_the_point_limit_widens_the_tolerance_and_says_so() -> None:
    curves, _ = _curves(_frame(), curve_max_points=5)
    discharge = curves[0]
    assert discharge["metrics"]["point_count"]["value"] <= 5
    assert discharge["metrics"]["voltage_tolerance"]["value"] > 0.0005
    assert discharge["metrics"]["max_interpolation_error"]["value"] <= discharge["metrics"]["voltage_tolerance"]["value"]
    assert "curve_tolerance_relaxed_to_point_limit" in discharge["quality"]["flags"]
    with pytest.raises(ValueError):
        AnalysisConfig(analysis_policy={"curve_max_points": 1})


def test_a_single_sample_segment_is_not_computable() -> None:
    frame = pl.DataFrame({"_source_row": [0, 1, 2], "test_time_s": [0.0, 10.0, 20.0], "current_a": [0.0, -1.0, 0.0],
                          "voltage_v": [3.6, 3.5, 3.55], "step_index": [1, 2, 3]})
    curves, _ = _curves(frame)
    assert curves[0]["applicability"]["status"] == "not_computable"
    assert curves[0]["applicability"]["reasons"] == ["fewer_than_two_distinct_capacity_values"]
    assert curves[0]["series"] == {}


def test_full_run_writes_curves(vendor_csv, tmp_path) -> None:
    result = bfl.analyze(vendor_csv, output_dir=tmp_path / "out", input_adapter="bds")
    evidence = json.loads(result.analysis_evidence_path.read_text(encoding="utf-8"))
    curves = [r for r in evidence["records"] if r["record_type"] == "response.capacity_voltage_curve"]
    assert curves
    for record in curves:
        if record["applicability"]["status"] == "applicable":
            assert len(record["series"]["capacity_ah"]) == record["metrics"]["point_count"]["value"]
            assert record["metrics"]["max_interpolation_error"]["value"] <= record["metrics"]["voltage_tolerance"]["value"]
    summary = json.loads(result.analysis_results_path.read_text(encoding="utf-8"))
    assert summary["retrieval"]["record_type_counts"]["response.capacity_voltage_curve"] == len(curves)
