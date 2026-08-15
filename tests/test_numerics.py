"""Tests for explicitly defined BFL numerical conventions."""

from __future__ import annotations

import numpy as np

from battery_feature_lab.analysis.numerics import (
    integrate_previous,
    reported_delta,
    weighted_quantile,
)
from battery_feature_lab.analysis.schema import metric


def test_previous_zoh_uses_left_sample_and_excludes_missing_duration() -> None:
    time = np.array([0.0, 1.0, 4.0, 5.0])
    values = np.array([2.0, np.nan, 4.0, 999.0])

    total, intervals = integrate_previous(time, values)

    assert total == 6.0
    assert intervals.included_duration_s == 2.0
    assert intervals.excluded_duration_s == 3.0
    assert intervals.values[-1] == 4.0


def test_weighted_quantile_is_inverted_cdf() -> None:
    result = weighted_quantile([1.0, 2.0, 10.0], [1.0, 8.0, 1.0], [0.05, 0.5, 0.95])
    np.testing.assert_array_equal(result, [1.0, 2.0, 10.0])


def test_bad_cumulative_column_is_rejected() -> None:
    value, flags = reported_delta([0.0, 1.0, 0.2, 1.2])
    assert value is None
    assert "reported_column_not_monotonic" in flags


def test_sampling_interval_outlier_and_duplicate_time_are_reported() -> None:
    _, intervals = integrate_previous(
        [0.0, 1.0, 1.0, 20.0, 21.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        sampling_interval_outlier_factor=5.0,
    )
    assert intervals.non_positive_interval_count == 1
    assert intervals.sampling_interval_outlier_count == 1
    assert intervals.sampling_interval_outlier_duration_s == 19.0
    assert intervals.max_sampling_interval_outlier_s == 19.0


def test_sustained_sampling_rate_changes_are_not_reported_as_gaps() -> None:
    durations = np.r_[np.full(10, 0.1), np.full(10, 1.0), np.full(10, 10.0)]
    time = np.r_[0.0, np.cumsum(durations)]

    _, intervals = integrate_previous(
        time,
        np.ones_like(time),
        sampling_interval_outlier_factor=5.0,
    )

    assert intervals.sampling_interval_outlier_count == 0
    assert intervals.sampling_interval_outlier_duration_s == 0.0
    assert intervals.max_sampling_interval_outlier_s is None


def test_missing_metric_value_cannot_claim_ok_status() -> None:
    result = metric(None, "V")

    assert result["status"] == "not_computable"
    assert result["reason"] == "value_unavailable"
