"""Relaxation/rest feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from battery_feature_lab.featurizers.base import BaseFeaturizer
from battery_feature_lab.featurizers.common import describe_array, duration_s
from battery_feature_lab.schemas import FeatureTable


class RelaxationFeaturizer(BaseFeaturizer):
    """Extract voltage relaxation features from rest segments."""

    name = "relaxation_features"

    def extract(self, frame: pd.DataFrame) -> FeatureTable:
        rows: list[dict[str, float | int | str]] = []
        rests = frame[frame["step_type"] == "rest"]
        if rests.empty:
            return FeatureTable(self.name, pd.DataFrame())
        for (cell_id, cycle_index, step_index), rest in rests.groupby(
            ["cell_id", "cycle_index", "step_index"], sort=True
        ):
            if len(rest) < 5:
                continue
            row = self._rest_features(rest)
            row.update({"cell_id": cell_id, "cycle_index": int(cycle_index), "step_index": int(step_index)})
            rows.append(row)
        return FeatureTable(self.name, pd.DataFrame(rows))

    def _rest_features(self, rest: pd.DataFrame) -> dict[str, float]:
        time = rest["time_s"].to_numpy(dtype=float)
        voltage = rest["voltage_v"].to_numpy(dtype=float)
        time = time - np.nanmin(time)
        order = np.argsort(time)
        time = time[order]
        voltage = voltage[order]
        valid = np.isfinite(time) & np.isfinite(voltage)
        time = time[valid]
        voltage = voltage[valid]
        features: dict[str, float] = {
            "rest_duration_s": duration_s(rest),
            "rest_voltage_initial_v": float(voltage[0]) if len(voltage) else float("nan"),
            "rest_voltage_final_v": float(voltage[-1]) if len(voltage) else float("nan"),
            "rest_voltage_delta_v": float(voltage[-1] - voltage[0]) if len(voltage) else float("nan"),
        }
        features.update(describe_array("rest_voltage_v", voltage))
        features.update(_interpolated_voltage_points(time, voltage))
        features.update(_slope_features(time, voltage))
        features.update(_exponential_fit(time, voltage))
        return features


def _interpolated_voltage_points(time: np.ndarray, voltage: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    if len(time) < 2:
        return result
    for seconds in (30, 60, 300, 1800):
        if seconds <= time.max():
            result[f"rest_voltage_at_{seconds}s_v"] = float(np.interp(seconds, time, voltage))
        else:
            result[f"rest_voltage_at_{seconds}s_v"] = float("nan")
    return result


def _slope_features(time: np.ndarray, voltage: np.ndarray) -> dict[str, float]:
    if len(time) < 3 or time.max() <= time.min():
        return {
            "rest_voltage_initial_slope_v_per_s": float("nan"),
            "rest_voltage_final_slope_v_per_s": float("nan"),
            "rest_voltage_linear_slope_v_per_s": float("nan"),
        }
    n = max(3, min(10, len(time) // 4))
    initial = np.polyfit(time[:n], voltage[:n], 1)[0]
    final = np.polyfit(time[-n:], voltage[-n:], 1)[0]
    linear = np.polyfit(time, voltage, 1)[0]
    return {
        "rest_voltage_initial_slope_v_per_s": float(initial),
        "rest_voltage_final_slope_v_per_s": float(final),
        "rest_voltage_linear_slope_v_per_s": float(linear),
    }


def _exponential_fit(time: np.ndarray, voltage: np.ndarray) -> dict[str, float]:
    """Fit v(t) = v_inf + a * exp(-t / tau)."""

    result = {
        "rest_exp_v_inf_v": float("nan"),
        "rest_exp_amplitude_v": float("nan"),
        "rest_exp_tau_s": float("nan"),
        "rest_exp_rmse_v": float("nan"),
    }
    if len(time) < 8 or time.max() <= 0:
        return result
    try:
        p0 = (float(voltage[-1]), float(voltage[0] - voltage[-1]), max(float(time.max()) / 3.0, 1.0))
        bounds = (
            (float(np.nanmin(voltage)) - 1.0, -5.0, 1e-6),
            (float(np.nanmax(voltage)) + 1.0, 5.0, max(float(time.max()) * 100.0, 1.0)),
        )
        params, _ = curve_fit(_exp_model, time, voltage, p0=p0, bounds=bounds, maxfev=10000)
        pred = _exp_model(time, *params)
        result["rest_exp_v_inf_v"] = float(params[0])
        result["rest_exp_amplitude_v"] = float(params[1])
        result["rest_exp_tau_s"] = float(params[2])
        result["rest_exp_rmse_v"] = float(np.sqrt(np.mean((pred - voltage) ** 2)))
    except (RuntimeError, ValueError, FloatingPointError):
        pass
    return result


def _exp_model(time: np.ndarray, v_inf: float, amplitude: float, tau: float) -> np.ndarray:
    return v_inf + amplitude * np.exp(-time / tau)
