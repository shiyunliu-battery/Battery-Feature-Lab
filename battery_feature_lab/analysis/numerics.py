"""Auditable numerical conventions not provided by the selected battery tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IntervalData:
    """Previous-sample, left-continuous intervals derived from time-series samples."""

    values: np.ndarray
    durations_s: np.ndarray
    valid: np.ndarray
    total_duration_s: float
    included_duration_s: float
    excluded_duration_s: float
    non_positive_interval_count: int
    median_positive_interval_s: float | None
    sampling_interval_outlier_count: int
    sampling_interval_outlier_duration_s: float
    max_sampling_interval_outlier_s: float | None


def previous_intervals(
    time_s: Any,
    values: Any,
    *,
    sampling_interval_outlier_factor: float = 5.0,
) -> IntervalData:
    """Assign sample ``x[i]`` to ``[t[i], t[i+1])`` without filling missing values."""

    time = np.asarray(time_s, dtype=float)
    signal = np.asarray(values, dtype=float)
    if time.ndim != 1 or signal.ndim != 1 or len(time) != len(signal):
        raise ValueError("time and values must be one-dimensional arrays of equal length")
    if len(time) < 2:
        empty = np.asarray([], dtype=float)
        return IntervalData(
            empty,
            empty,
            np.asarray([], dtype=bool),
            0.0,
            0.0,
            0.0,
            0,
            None,
            0,
            0.0,
            None,
        )

    durations = np.diff(time)
    held = signal[:-1]
    positive = np.isfinite(durations) & (durations > 0)
    valid = positive & np.isfinite(held)
    positive_dt = durations[positive]
    median = float(np.median(positive_dt)) if positive_dt.size else None
    outliers = _isolated_interval_outliers(
        durations,
        positive,
        sampling_interval_outlier_factor,
    )
    total = float(np.sum(durations[positive]))
    included = float(np.sum(durations[valid]))
    return IntervalData(
        values=held,
        durations_s=durations,
        valid=valid,
        total_duration_s=total,
        included_duration_s=included,
        excluded_duration_s=max(total - included, 0.0),
        non_positive_interval_count=int(np.sum(np.isfinite(durations) & (durations <= 0))),
        median_positive_interval_s=median,
        sampling_interval_outlier_count=int(np.sum(outliers)),
        sampling_interval_outlier_duration_s=float(np.sum(durations[outliers])),
        max_sampling_interval_outlier_s=(
            float(np.max(durations[outliers])) if np.any(outliers) else None
        ),
    )


def _isolated_interval_outliers(
    durations: np.ndarray,
    positive: np.ndarray,
    factor: float,
) -> np.ndarray:
    """Flag an interval only when it exceeds both adjacent sampling regimes.

    Battery cyclers commonly switch between sustained 0.1 s, 1 s, 10 s, and
    60 s logging blocks. A global-median rule mislabels those blocks as gaps.
    Comparing with the nearest positive interval on each side preserves rate
    transitions. The result is a sampling-interval outlier, not proof that data
    are missing.
    """

    count = len(durations)
    if count < 2:
        return np.zeros(count, dtype=bool)
    indexes = np.arange(count)
    previous_index = np.maximum.accumulate(np.where(positive, indexes, -1))
    previous_index = np.r_[-1, previous_index[:-1]]
    next_index = np.minimum.accumulate(np.where(positive, indexes, count)[::-1])[::-1]
    next_index = np.r_[next_index[1:], count]

    previous = np.full(count, np.nan, dtype=float)
    following = np.full(count, np.nan, dtype=float)
    has_previous = previous_index >= 0
    has_following = next_index < count
    previous[has_previous] = durations[previous_index[has_previous]]
    following[has_following] = durations[next_index[has_following]]
    reference = np.fmax(previous, following)
    return positive & np.isfinite(reference) & (durations > factor * reference)


def weighted_quantile(values: Any, weights: Any, quantiles: Any) -> np.ndarray:
    """Weighted inverted-CDF quantiles with positive finite weights."""

    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    q = np.atleast_1d(np.asarray(quantiles, dtype=float))
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return np.full(q.shape, np.nan, dtype=float)
    order = np.argsort(x[valid], kind="mergesort")
    sorted_x = x[valid][order]
    sorted_w = w[valid][order]
    cumulative = np.cumsum(sorted_w)
    targets = np.clip(q, 0.0, 1.0) * cumulative[-1]
    indexes = np.searchsorted(cumulative, targets, side="left")
    return sorted_x[np.clip(indexes, 0, len(sorted_x) - 1)]


def weighted_summary(intervals: IntervalData) -> dict[str, float | None]:
    """Return duration-weighted exposure statistics for valid intervals."""

    values = intervals.values[intervals.valid]
    weights = intervals.durations_s[intervals.valid]
    if not len(values) or float(np.sum(weights)) <= 0:
        return {name: None for name in ("min", "max", "mean", "rms", "q05", "q50", "q95", "q99")}
    quantiles = weighted_quantile(values, weights, [0.05, 0.5, 0.95, 0.99])
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.average(values, weights=weights)),
        "rms": float(np.sqrt(np.average(np.square(values), weights=weights))),
        "q05": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "q99": float(quantiles[3]),
    }


def integrate_previous(
    time_s: Any,
    values: Any,
    *,
    scale: float = 1.0,
    sampling_interval_outlier_factor: float = 5.0,
) -> tuple[float | None, IntervalData]:
    """Integrate a sampled signal using previous-sample ZOH."""

    intervals = previous_intervals(
        time_s,
        values,
        sampling_interval_outlier_factor=sampling_interval_outlier_factor,
    )
    if not np.any(intervals.valid):
        return None, intervals
    total = np.sum(intervals.values[intervals.valid] * intervals.durations_s[intervals.valid])
    return float(total * scale), intervals


def reported_delta(
    values: Any, *, monotonic_tolerance: float = 1e-10
) -> tuple[float | None, list[str]]:
    """Validate a reported cumulative column and return its endpoint delta."""

    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    flags: list[str] = []
    if finite.size < 2:
        return None, ["reported_column_has_fewer_than_two_finite_values"]
    diffs = np.diff(finite)
    if np.any(diffs < -monotonic_tolerance):
        flags.append("reported_column_not_monotonic")
    delta = float(finite[-1] - finite[0])
    if delta < -monotonic_tolerance:
        flags.append("reported_column_negative_delta")
    if flags:
        return None, flags
    return max(delta, 0.0), flags


def finite_coverage(time_s: Any, *signals: Any) -> float:
    """Fraction of positive time duration covered by all supplied held signals."""

    time = np.asarray(time_s, dtype=float)
    if len(time) < 2:
        return 0.0
    dt = np.diff(time)
    positive = np.isfinite(dt) & (dt > 0)
    if not np.any(positive):
        return 0.0
    valid = positive.copy()
    for signal in signals:
        values = np.asarray(signal, dtype=float)
        valid &= np.isfinite(values[:-1])
    return float(np.sum(dt[valid]) / np.sum(dt[positive]))


def mostly_monotone(values: Any, *, increasing: bool, fraction: float = 0.9) -> bool:
    """Return whether finite adjacent differences mostly have the requested direction."""

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 3:
        return False
    diffs = np.diff(data)
    if increasing:
        return bool(np.mean(diffs >= -1e-12) >= fraction)
    return bool(np.mean(diffs <= 1e-12) >= fraction)


def json_number(value: Any) -> float | int | None:
    """Convert NumPy scalars to finite JSON numbers."""

    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number
