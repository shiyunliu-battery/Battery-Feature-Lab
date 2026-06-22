"""Unit and sign normalization."""

from __future__ import annotations

import pandas as pd

from battery_feature_lab.schemas import ReaderConfig


def normalize_units_and_sign(frame: pd.DataFrame, config: ReaderConfig) -> pd.DataFrame:
    """Normalize time, capacity, SOC and current sign convention."""

    normalized = frame.copy()
    if "time_s" in normalized.columns and config.time_unit.lower() in {"h", "hr", "hour", "hours"}:
        normalized["time_s"] = normalized["time_s"] * 3600.0
    elif "time_s" in normalized.columns and config.time_unit.lower() in {"min", "minute", "minutes"}:
        normalized["time_s"] = normalized["time_s"] * 60.0

    if config.capacity_unit.lower() in {"mah", "milliamp_hour", "milliamp-hours"}:
        for column in ("charge_capacity_ah", "discharge_capacity_ah"):
            if column in normalized.columns:
                normalized[column] = normalized[column] / 1000.0

    if "soc" in normalized.columns and config.soc_unit.lower() in {"percent", "%"}:
        normalized["soc"] = normalized["soc"] / 100.0

    if "current_a" in normalized.columns and not config.positive_current_is_charge:
        normalized["current_a"] = -normalized["current_a"]

    return normalized
