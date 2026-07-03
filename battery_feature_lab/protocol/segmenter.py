"""Segment a normalized time series into primitive protocol steps.

The approach is adapted from established open-source tools, using only BFL's
existing dependencies (numpy/pandas):

- The step-table philosophy of cellpy (jepegit/cellpy): group contiguous
  samples that share a current mode and categorise each run (rest / CC / CV /
  pulse / dynamic).
- The pulse/relaxation descriptors of ampworks (NatLabRockies/ampworks) and
  ionworkspipeline inform the per-segment voltage descriptors.

Nothing here is a hard dependency on those packages; the reliable *ideas* are
reimplemented so BFL stays lean.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from battery_feature_lab.schemas import ProtocolConfig

# Segment step labels.
REST = "rest"
CC_CHARGE = "cc_charge"
CC_DISCHARGE = "cc_discharge"
CV_CHARGE = "cv_charge"
CV_DISCHARGE = "cv_discharge"
PULSE_CHARGE = "pulse_charge"
PULSE_DISCHARGE = "pulse_discharge"
DYNAMIC = "dynamic"


def segment_cycle(
    cycle_frame: pd.DataFrame,
    config: ProtocolConfig,
    nominal_capacity_ah: float | None = None,
) -> list[dict[str, Any]]:
    """Split one cycle into ordered primitive segments with descriptors.

    Returns a list of segment dicts in time order. Each segment carries its
    time/voltage/current descriptors and a ``step_label``. Pulse relabelling
    (which needs neighbour context) is applied here as a second pass.
    """

    frame = cycle_frame.sort_values("time_s")
    time = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(frame["voltage_v"], errors="coerce").to_numpy(dtype=float)
    current = pd.to_numeric(frame["current_a"], errors="coerce").to_numpy(dtype=float)
    index = frame.index.to_numpy()

    valid = np.isfinite(time) & np.isfinite(current)
    time, voltage, current, index = time[valid], voltage[valid], current[valid], index[valid]
    if time.size == 0:
        return []

    modes = _current_mode(current, config.rest_threshold_a)
    # First split active operation from rest. Splitting on current sign here would
    # turn a sign-changing drive cycle into dozens of one-point CC segments.
    runs = _contiguous_runs((modes != 0).astype(int))

    segments: list[dict[str, Any]] = []
    for start, stop in runs:
        active = modes[start] != 0
        sl_t, sl_v, sl_i, sl_idx = time[start:stop], voltage[start:stop], current[start:stop], index[start:stop]
        if not active:
            segments.append(_segment_record(REST, sl_t, sl_v, sl_i, sl_idx, nominal_capacity_ah))
            continue
        nonzero_modes = modes[start:stop]
        if np.any(nonzero_modes > 0) and np.any(nonzero_modes < 0):
            segments.append(
                _segment_record(DYNAMIC, sl_t, sl_v, sl_i, sl_idx, nominal_capacity_ah)
            )
            continue
        mode = 1 if np.any(nonzero_modes > 0) else -1
        # Non-zero-current run: may be pure CC, pure CV, CC+CV, or dynamic.
        for label, lo, hi in _label_current_run(sl_t, sl_v, sl_i, mode, config):
            segments.append(
                _segment_record(label, sl_t[lo:hi], sl_v[lo:hi], sl_i[lo:hi], sl_idx[lo:hi], nominal_capacity_ah)
            )

    _relabel_pulses(segments, config)
    for position, segment in enumerate(segments):
        segment["segment_index"] = position
    return segments


def _current_mode(current: np.ndarray, rest_threshold_a: float) -> np.ndarray:
    """Map each sample to +1 (charge), -1 (discharge) or 0 (rest)."""

    mode = np.zeros(current.shape, dtype=int)
    mode[current > rest_threshold_a] = 1
    mode[current < -rest_threshold_a] = -1
    return mode


def _contiguous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, stop) index pairs for contiguous equal-value runs."""

    if values.size == 0:
        return []
    change_points = np.flatnonzero(np.diff(values)) + 1
    bounds = [0, *change_points.tolist(), values.size]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _label_current_run(
    time: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    mode: int,
    config: ProtocolConfig,
) -> list[tuple[str, int, int]]:
    """Label a single non-zero-current run, splitting a CC head from a CV tail.

    Returns ``(label, lo, hi)`` sub-slices covering the whole run in order.
    """

    n = current.size
    abs_current = np.abs(current)
    peak = float(np.nanmax(abs_current)) if n else 0.0

    # Constant-voltage tail candidate: contiguous suffix where current has
    # tapered below a fraction of the run peak. It is accepted only if voltage
    # is actually stable over that suffix.
    cv_start = n
    if peak > 0:
        tapered = abs_current < config.cv_current_fraction * peak
        i = n - 1
        while i >= 0 and tapered[i]:
            cv_start = i
            i -= 1

    cc_len = cv_start
    cv_len = n - cv_start
    if cv_len >= config.min_segment_points and not _is_voltage_stable(
        time[cv_start:], voltage[cv_start:], config
    ):
        cv_start = n
        cc_len = n
        cv_len = 0
    charge = mode > 0
    cc_label = CC_CHARGE if charge else CC_DISCHARGE
    cv_label = CV_CHARGE if charge else CV_DISCHARGE

    # If the run's current is neither near-constant nor a clean monotone taper, it
    # is a dynamic (drive-cycle / DST) load rather than a CC/CV step.
    if cv_len < config.min_segment_points and not _is_constant(
        abs_current, config.cc_current_cv_tol
    ):
        return [(DYNAMIC, 0, n)]

    if cc_len < config.min_segment_points:
        return [(cv_label if cv_len else cc_label, 0, n)]
    if cv_len < config.min_segment_points:
        return [(cc_label, 0, n)]
    return [(cc_label, 0, cc_len), (cv_label, cc_len, n)]


def _is_constant(values: np.ndarray, tol: float) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return True
    mean = float(np.mean(finite))
    if mean == 0:
        return True
    return float(np.std(finite)) / abs(mean) <= tol


def _is_monotone_nonincreasing(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return True
    # Allow small non-monotone noise: at least 80% of steps are non-increasing.
    diffs = np.diff(finite)
    return float(np.mean(diffs <= 1e-12)) >= 0.8


def _is_voltage_stable(
    time: np.ndarray,
    voltage: np.ndarray,
    config: ProtocolConfig,
) -> bool:
    valid = np.isfinite(time) & np.isfinite(voltage)
    t, v = time[valid], voltage[valid]
    if v.size < config.min_segment_points:
        return False
    voltage_range = float(np.max(v) - np.min(v))
    slope = abs(_linear_slope(t, v))
    return (
        voltage_range <= config.cv_voltage_tolerance_v
        and np.isfinite(slope)
        and slope <= config.cv_voltage_slope_tolerance_v_per_s
    )


def _relabel_pulses(segments: list[dict[str, Any]], config: ProtocolConfig) -> None:
    """Relabel short current segments adjacent to a rest as pulses (GITT/HPPC)."""

    for position, segment in enumerate(segments):
        label = segment["step_label"]
        if label not in (CC_CHARGE, CC_DISCHARGE):
            continue
        if segment["duration_s"] > config.pulse_max_duration_s:
            continue
        adjacent_rests = [
            neighbour
            for neighbour in (
                segments[position - 1] if position > 0 else None,
                segments[position + 1] if position + 1 < len(segments) else None,
            )
            if neighbour is not None and neighbour["step_label"] == REST
        ]
        has_qualifying_rest = any(
            rest["duration_s"] >= config.pulse_rest_min_duration_s
            and rest["duration_s"]
            >= segment["duration_s"] * config.pulse_rest_duration_ratio
            for rest in adjacent_rests
        )
        if has_qualifying_rest:
            segment["step_label"] = PULSE_CHARGE if label == CC_CHARGE else PULSE_DISCHARGE


def _segment_record(
    label: str,
    time: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    index: np.ndarray,
    nominal_capacity_ah: float | None,
) -> dict[str, Any]:
    duration = float(time[-1] - time[0]) if time.size else 0.0
    v_valid = voltage[np.isfinite(voltage)]
    start_v = float(voltage[0]) if voltage.size and np.isfinite(voltage[0]) else float("nan")
    end_v = float(voltage[-1]) if voltage.size and np.isfinite(voltage[-1]) else float("nan")
    max_abs_current = float(np.nanmax(np.abs(current))) if current.size else 0.0
    mean_current = float(np.nanmean(current)) if current.size else 0.0
    c_rate = (
        max_abs_current / nominal_capacity_ah
        if nominal_capacity_ah and nominal_capacity_ah > 0
        else float("nan")
    )
    return {
        "step_label": label,
        "start_time_s": float(time[0]) if time.size else float("nan"),
        "end_time_s": float(time[-1]) if time.size else float("nan"),
        "duration_s": duration,
        "n_points": int(time.size),
        "start_voltage_v": start_v,
        "end_voltage_v": end_v,
        "delta_voltage_v": end_v - start_v,
        "voltage_range_v": float(np.nanmax(v_valid) - np.nanmin(v_valid)) if v_valid.size else float("nan"),
        "voltage_slope_v_per_s": _linear_slope(time, voltage),
        "mean_current_a": mean_current,
        "max_abs_current_a": max_abs_current,
        "current_sign": int(np.sign(mean_current)),
        "c_rate": c_rate,
        "start_row_index": int(index[0]) if index.size else -1,
        "end_row_index": int(index[-1]) if index.size else -1,
        "source_row_indices": [int(value) for value in index.tolist()],
    }


def _linear_slope(time: np.ndarray, voltage: np.ndarray) -> float:
    valid = np.isfinite(time) & np.isfinite(voltage)
    t, v = time[valid], voltage[valid]
    if t.size < 2 or t.max() <= t.min():
        return float("nan")
    return float(np.polyfit(t - t.min(), v, 1)[0])
