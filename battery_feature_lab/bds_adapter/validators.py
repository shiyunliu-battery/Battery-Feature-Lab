"""Validation helpers for normalized battery data."""

from __future__ import annotations

import pandas as pd


def validate_timeseries(frame: pd.DataFrame) -> None:
    """Validate the minimum canonical time-series schema."""

    required = {"cell_id", "cycle_index", "time_s", "voltage_v", "current_a", "step_type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required canonical columns: {missing}")

    if frame.empty:
        raise ValueError("Input data is empty after normalization.")

    for column in ("time_s", "voltage_v", "current_a"):
        if frame[column].isna().all():
            raise ValueError(f"Column {column} contains only missing values.")

    valid_steps = {"charge", "discharge", "rest"}
    bad_steps = sorted(set(frame["step_type"].dropna()) - valid_steps)
    if bad_steps:
        raise ValueError(f"Unexpected step_type values: {bad_steps}")
