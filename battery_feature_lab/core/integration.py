"""Integration helpers for capacity and energy."""

from __future__ import annotations

import numpy as np
import pandas as pd

# numpy>=2.0 renamed np.trapz to np.trapezoid (np.trapz is deprecated). Resolve once so the
# package works across the declared numpy range (>=1.23) without per-call branching.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of ``y`` over ``x``, compatible with numpy 1.x and 2.x."""

    return float(_trapezoid(y, x))


def elapsed_hours(time_s: pd.Series) -> np.ndarray:
    """Return positive time deltas in hours."""

    time = pd.to_numeric(time_s, errors="coerce").to_numpy(dtype=float)
    if len(time) == 0:
        return np.array([])
    delta = np.diff(time, prepend=time[0])
    delta[~np.isfinite(delta)] = 0.0
    delta[delta < 0] = 0.0
    return delta / 3600.0


def integrate_capacity_ah(frame: pd.DataFrame, signed: bool = False) -> float:
    """Integrate current over time and return capacity in Ah."""

    if frame.empty:
        return float("nan")
    dt_h = elapsed_hours(frame["time_s"])
    current = pd.to_numeric(frame["current_a"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    current_used = current if signed else np.abs(current)
    return float(np.nansum(current_used * dt_h))


def integrate_energy_wh(frame: pd.DataFrame, signed: bool = False) -> float:
    """Integrate voltage-current power over time and return energy in Wh."""

    if frame.empty:
        return float("nan")
    dt_h = elapsed_hours(frame["time_s"])
    current = pd.to_numeric(frame["current_a"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    voltage = pd.to_numeric(frame["voltage_v"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    power = voltage * current
    power_used = power if signed else np.abs(power)
    return float(np.nansum(power_used * dt_h))


def capacity_from_columns_or_integral(frame: pd.DataFrame, capacity_column: str) -> float:
    """Use cumulative capacity column when present, otherwise integrate current."""

    if capacity_column in frame.columns and frame[capacity_column].notna().sum() >= 2:
        values = pd.to_numeric(frame[capacity_column], errors="coerce").dropna()
        if values.empty:
            return integrate_capacity_ah(frame)
        delta = float(values.iloc[-1] - values.iloc[0])
        if abs(delta) > 0:
            return abs(delta)
        return float(values.max() - values.min())
    return integrate_capacity_ah(frame)


def cumulative_capacity_ah(frame: pd.DataFrame) -> np.ndarray:
    """Return absolute cumulative capacity within a frame."""

    dt_h = elapsed_hours(frame["time_s"])
    current = pd.to_numeric(frame["current_a"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.cumsum(np.abs(current) * dt_h)
