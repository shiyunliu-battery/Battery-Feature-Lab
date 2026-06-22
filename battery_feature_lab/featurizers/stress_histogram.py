"""Usage-stress and dynamic-operation feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_feature_lab.core.integration import elapsed_hours
from battery_feature_lab.featurizers.base import BaseFeaturizer
from battery_feature_lab.featurizers.common import describe_array, safe_ratio
from battery_feature_lab.schemas import FeatureTable


class StressHistogramFeaturizer(BaseFeaturizer):
    """Extract field-data style stress features from historical usage."""

    name = "stress_features"

    def extract(self, frame: pd.DataFrame) -> FeatureTable:
        rows: list[dict[str, float | int | str]] = []
        for cell_id, cell in frame.groupby("cell_id", sort=True):
            rows.append(self._cell_features(cell_id, cell))
        return FeatureTable(self.name, pd.DataFrame(rows))

    def _cell_features(self, cell_id: str, cell: pd.DataFrame) -> dict[str, float | int | str]:
        time_span_s = float(cell["time_s"].max() - cell["time_s"].min())
        dt_h = elapsed_hours(cell["time_s"])
        current = cell["current_a"].to_numpy(dtype=float)
        voltage = cell["voltage_v"].to_numpy(dtype=float)
        abs_current = np.abs(current)
        nominal = self.config.nominal_capacity_ah
        c_rate = abs_current / nominal if nominal else np.full(len(cell), np.nan)
        soc = _soc_vector(cell, nominal)
        temp = cell["temperature_c"].to_numpy(dtype=float) if "temperature_c" in cell else np.full(len(cell), np.nan)
        rest_mask = cell["step_type"].to_numpy() == "rest"
        high_soc_mask = soc >= self.config.high_soc_level
        charge_mask = current > 0
        discharge_mask = current < 0

        features: dict[str, float | int | str] = {
            "cell_id": cell_id,
            "n_points": int(len(cell)),
            "observed_time_span_s": time_span_s,
            "calendar_time_h": float(np.nansum(dt_h)),
            "throughput_ah": float(np.nansum(abs_current * dt_h)),
            "equivalent_full_cycles": safe_ratio(float(np.nansum(abs_current * dt_h)) / 2.0, nominal or float("nan")),
            "high_soc_level_used": float(self.config.high_soc_level),
            "high_soc_rest_fraction": _weighted_fraction(rest_mask & high_soc_mask, dt_h),
            "rest_fraction": _weighted_fraction(rest_mask, dt_h),
            "charge_fraction": _weighted_fraction(charge_mask, dt_h),
            "discharge_fraction": _weighted_fraction(discharge_mask, dt_h),
            "max_instant_discharge_c_rate": float(np.nanmax(np.where(discharge_mask, c_rate, np.nan))),
            "current_variance_a2": float(np.nanvar(current, ddof=1)) if len(current) > 1 else 0.0,
            "c_rate_variance": float(np.nanvar(c_rate, ddof=1)) if len(c_rate) > 1 else float("nan"),
            "low_frequency_current_power": _low_frequency_power(cell["time_s"].to_numpy(dtype=float), current),
        }
        features.update(describe_array("soc", soc))
        features.update(describe_array("voltage_v", voltage))
        features.update(describe_array("current_a", current))
        features.update(describe_array("abs_current_a", abs_current))
        features.update(describe_array("c_rate", c_rate))
        features.update(describe_array("temperature_c", temp))
        features.update(_histogram_features("soc", soc, bins=np.linspace(0, 1, self.config.histogram_bins + 1)))
        features.update(_histogram_features("voltage_v", voltage, bins=self.config.histogram_bins))
        features.update(_histogram_features("temperature_c", temp, bins=self.config.histogram_bins))
        features.update(_histogram_features("c_rate", c_rate, bins=self.config.histogram_bins))
        return features


def _soc_vector(cell: pd.DataFrame, nominal_capacity_ah: float | None) -> np.ndarray:
    if "soc" in cell.columns and cell["soc"].notna().sum() > 0:
        soc = cell["soc"].to_numpy(dtype=float)
        if np.nanmax(soc) > 1.5:
            soc = soc / 100.0
        return np.clip(soc, 0.0, 1.0)
    if nominal_capacity_ah and "discharge_capacity_ah" in cell.columns:
        qd = pd.to_numeric(cell["discharge_capacity_ah"], errors="coerce").to_numpy(dtype=float)
        qd = qd - np.nanmin(qd)
        return np.clip(1.0 - qd / nominal_capacity_ah, 0.0, 1.0)
    voltage = cell["voltage_v"].to_numpy(dtype=float)
    vmin, vmax = np.nanmin(voltage), np.nanmax(voltage)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.full(len(cell), np.nan)
    return np.clip((voltage - vmin) / (vmax - vmin), 0.0, 1.0)


def _weighted_fraction(mask: np.ndarray, weights: np.ndarray) -> float:
    total = np.nansum(weights)
    if total <= 0:
        return float("nan")
    return float(np.nansum(weights[np.asarray(mask, dtype=bool)]) / total)


def _histogram_features(prefix: str, values: np.ndarray, bins: int | np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    result: dict[str, float] = {}
    if len(values) == 0:
        return result
    counts, edges = np.histogram(values, bins=bins)
    total = counts.sum()
    if total == 0:
        return result
    fractions = counts / total
    for idx, fraction in enumerate(fractions):
        result[f"{prefix}_hist_bin_{idx}_fraction"] = float(fraction)
        result[f"{prefix}_hist_bin_{idx}_low"] = float(edges[idx])
        result[f"{prefix}_hist_bin_{idx}_high"] = float(edges[idx + 1])
    return result


def _low_frequency_power(time_s: np.ndarray, current_a: np.ndarray) -> float:
    """Estimate current-profile power below 0.1 Hz for SHAP-style dynamic features."""

    valid = np.isfinite(time_s) & np.isfinite(current_a)
    time_s = time_s[valid]
    current_a = current_a[valid]
    if len(time_s) < 16 or time_s.max() <= time_s.min():
        return float("nan")
    order = np.argsort(time_s)
    time_s = time_s[order]
    current_a = current_a[order] - np.mean(current_a)
    dt = np.median(np.diff(time_s))
    if not np.isfinite(dt) or dt <= 0:
        return float("nan")
    freqs = np.fft.rfftfreq(len(current_a), d=dt)
    spectrum = np.abs(np.fft.rfft(current_a)) ** 2
    mask = (freqs > 0) & (freqs <= 0.1)
    if not mask.any():
        return 0.0
    return float(np.sum(spectrum[mask]) / max(np.sum(spectrum), 1e-12))
