"""Cycle and step inference helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def infer_step_type(current_a: pd.Series, rest_threshold_a: float = 1e-4) -> pd.Series:
    """Infer charge/discharge/rest from current sign."""

    current = pd.to_numeric(current_a, errors="coerce").fillna(0.0)
    values = np.where(
        current > rest_threshold_a,
        "charge",
        np.where(current < -rest_threshold_a, "discharge", "rest"),
    )
    return pd.Series(values, index=current_a.index, dtype="object")


def infer_cycle_index(frame: pd.DataFrame) -> pd.Series:
    """Infer cycle index from charge-to-discharge transitions.

    This fallback is intended for exports without explicit cycle numbers. If the input has
    protocol-specific semantics, a BDS adapter should provide exact cycle IDs instead.
    """

    current = pd.to_numeric(frame["current_a"], errors="coerce").fillna(0.0)
    sign = np.sign(current)
    sign[np.abs(current) < 1e-4] = 0
    cycle = np.zeros(len(frame), dtype=int)
    current_cycle = 1
    seen_discharge = False
    previous_non_rest = 0
    for i, value in enumerate(sign):
        if value < 0:
            seen_discharge = True
        if value > 0 and seen_discharge and previous_non_rest <= 0:
            current_cycle += 1
            seen_discharge = False
        if value != 0:
            previous_non_rest = value
        cycle[i] = current_cycle
    return pd.Series(cycle, index=frame.index, dtype="int64")


def iter_cycles(frame: pd.DataFrame):
    """Yield ``(cell_id, cycle_index, cycle_frame)`` sorted by cycle."""

    sorted_frame = frame.sort_values(["cell_id", "cycle_index", "time_s"])
    for (cell_id, cycle_index), group in sorted_frame.groupby(["cell_id", "cycle_index"], sort=True):
        yield cell_id, int(cycle_index), group.copy()
