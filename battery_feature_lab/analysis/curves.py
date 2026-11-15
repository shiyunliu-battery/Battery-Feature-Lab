"""Voltage against capacity for each charge and discharge phase segment, kept as a polyline with a stated error bound.

A ``response.capacity_voltage_curve`` record holds, for one phase segment, the voltage as a function of the
capacity moved since the segment's first record (trapezoid rule on the current magnitude over test time).
The samples are thinned with a vertical-deviation Ramer–Douglas–Peucker pass: a sample is kept only where
dropping it would move the linear interpolation by more than ``curve_voltage_tolerance_v``. Because both the
samples' interpolation and the kept polyline are piecewise linear with breakpoints at samples, the bound
checked at every sample holds everywhere, so two curves can be compared on any capacity grid with an error
of at most the sum of their tolerances. When a curve would keep more than ``curve_max_points`` points, the
tolerance is doubled until it does not, and the record says so.

The curve is a reading aid for comparisons between segments, specimens or files; it adds nothing between
samples and capacity is not a state of charge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from battery_feature_lab._version import PROVIDER_VERSION
from battery_feature_lab.analysis.schema import (
    ANALYSIS_POLICY_VERSION,
    AnalysisConfig,
    make_record,
    metric,
)

METHOD = "trapezoid_capacity_voltage_polyline_v1"
REFERENCES = [
    "https://doi.org/10.1016/S0146-664X(72)80017-0",  # Ramer (1972)
    "https://doi.org/10.3138/FM57-6770-U75U-7727",  # Douglas and Peucker (1973)
]
PHASES = ("charge", "discharge")


def simplify_polyline(x: np.ndarray, y: np.ndarray, tolerance: float) -> np.ndarray:
    """Indices of the points kept so that the polyline through them differs from the polyline through all
    points by at most ``tolerance`` in ``y`` at every point. ``x`` must be strictly increasing. Iterative,
    deterministic, always keeps the first and the last point."""

    n = len(x)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        inner = slice(i + 1, j)
        line = y[i] + (y[j] - y[i]) * (x[inner] - x[i]) / (x[j] - x[i])
        deviation = np.abs(y[inner] - line)
        k = int(np.argmax(deviation))
        if deviation[k] > tolerance:
            m = i + 1 + k
            keep[m] = True
            stack.append((m, j))
            stack.append((i, m))
    return np.flatnonzero(keep)


def capacity_voltage_samples(time: np.ndarray, current: np.ndarray, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Capacity from the first record (Ah, trapezoid on |I|) and voltage at each sample with a finite voltage,
    with one sample per capacity value (the last), and the segment's total capacity.

    An interval with a non-finite current or time, or a non-positive duration, adds no capacity; it is not
    filled."""

    absolute = np.abs(current)
    dt = np.diff(time)
    valid = np.isfinite(dt) & (dt > 0) & np.isfinite(absolute[:-1]) & np.isfinite(absolute[1:])
    increments = np.where(valid, 0.5 * (absolute[:-1] + absolute[1:]) * np.where(valid, dt, 0.0), 0.0) / 3600.0
    capacity = np.r_[0.0, np.cumsum(increments)]
    finite = np.isfinite(voltage)
    q, v = capacity[finite], voltage[finite]
    if len(q) == 0:
        return q, v, float(capacity[-1]) if len(capacity) else 0.0
    last_of_each = np.r_[np.diff(q) > 0, True]
    return q[last_of_each], v[last_of_each], float(capacity[-1])


def analyze_capacity_voltage_curves(
    frame: pl.DataFrame,
    phases: list[dict[str, Any]],
    *,
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
) -> list[dict[str, Any]]:
    """One ``response.capacity_voltage_curve`` record per charge or discharge phase segment."""

    tolerance_policy = config.policy("curve_voltage_tolerance_v")
    max_points = int(config.policy("curve_max_points"))
    records = []
    for segment in phases:
        if segment["phase"] not in PHASES:
            continue
        sub = frame.slice(segment["start"], segment["end"] - segment["start"])
        time = sub["test_time_s"].cast(pl.Float64, strict=False).to_numpy()
        current = sub["current_a"].cast(pl.Float64, strict=False).to_numpy()
        voltage = sub["voltage_v"].cast(pl.Float64, strict=False).to_numpy()
        q, v, total = capacity_voltage_samples(time, current, voltage)
        flags: list[str] = []
        reasons: list[str] = []
        series: dict[str, Any] = {}
        tolerance = tolerance_policy
        if len(q) < 2:
            reasons.append("fewer_than_two_distinct_capacity_values" if len(q) else "no_finite_voltage")
            kept = np.arange(0)
            error = None
        else:
            kept = simplify_polyline(q, v, tolerance)
            while len(kept) > max_points:
                tolerance *= 2.0
                kept = simplify_polyline(q, v, tolerance)
            if tolerance != tolerance_policy:
                flags.append("curve_tolerance_relaxed_to_point_limit")
            error = float(np.max(np.abs(np.interp(q, q[kept], v[kept]) - v)))
            series = {"capacity_ah": [float(x) for x in q[kept]], "voltage_v": [float(x) for x in v[kept]]}
        computable = not reasons
        records.append(
            make_record(
                record_id=f"response.capacity_voltage_curve:{cycle_id}:{segment['segment_index']}",
                record_type="response.capacity_voltage_curve",
                cell_id=cell_id,
                cycle_scope=cycle_id,
                source_intervals=[segment["source_interval"]],
                attributes={
                    "phase": segment["phase"],
                    "source_step_index": segment.get("source_step_index"),
                    "phase_segment_record_id": f"operation.phase_segment:{cycle_id}:{segment['segment_index']}",
                    "capacity_axis": "capacity moved since the first record of the segment (Ah); not a state of charge",
                },
                metrics={
                    "capacity": metric(total if computable else None, "Ah", reason=reasons[0] if reasons else None),
                    "point_count": metric(len(kept) if computable else None, "1", reason=reasons[0] if reasons else None),
                    "distinct_capacity_samples": metric(len(q), "1"),
                    "voltage_tolerance": metric(tolerance if computable else None, "V", reason=reasons[0] if reasons else None),
                    "max_interpolation_error": metric(error, "V", reason=reasons[0] if reasons else None),
                },
                series=series,
                provider="BFL",
                method_name=METHOD,
                provider_version=PROVIDER_VERSION,
                parameters={
                    "analysis_policy_version": ANALYSIS_POLICY_VERSION,
                    "voltage_tolerance_v": tolerance_policy,
                    "max_points": max_points,
                    "capacity_integration": (
                        "trapezoid rule on the current magnitude over test time from the first record of the segment; "
                        "an interval with a missing sample or a non-positive duration adds nothing"
                    ),
                    "duplicate_capacity": "last sample of each capacity value",
                    "simplification": (
                        "vertical-deviation Ramer-Douglas-Peucker; the kept polyline differs from the samples' linear "
                        "interpolation by at most the tolerance at every sample, hence everywhere"
                    ),
                    "tolerance_relaxation": "doubled until the point count is at most max_points",
                },
                references=REFERENCES,
                applicability_status="applicable" if computable else "not_computable",
                applicability_reasons=reasons,
                quality_status="warning" if flags else "ok",
                quality_flags=flags,
                interpretation_limits=[
                    "Capacity is counted from the first record of the segment; it is not a state of charge.",
                    "The curve is the samples' linear interpolation within voltage_tolerance; it adds nothing between samples.",
                ],
            )
        )
    return records
