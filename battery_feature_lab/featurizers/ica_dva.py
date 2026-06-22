"""ICA/DVA curve and peak feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

from battery_feature_lab.core.integration import cumulative_capacity_ah, trapezoid
from battery_feature_lab.core.resampling import resample_xy
from battery_feature_lab.core.smoothing import safe_gradient, safe_savgol
from battery_feature_lab.featurizers.base import BaseFeaturizer
from battery_feature_lab.featurizers.common import describe_array
from battery_feature_lab.schemas import FeatureTable


class ICADVAFeaturizer(BaseFeaturizer):
    """Extract incremental capacity and differential voltage features per cycle."""

    name = "ica_dva_features"

    def extract(self, frame: pd.DataFrame) -> FeatureTable:
        rows: list[dict[str, float | int | str]] = []
        for (cell_id, cycle_index, step_type), segment in frame.groupby(
            ["cell_id", "cycle_index", "step_type"], sort=True
        ):
            if step_type not in {"charge", "discharge"} or len(segment) < self.config.min_points_for_curve:
                continue
            curve = self._curve_features(segment)
            if curve is None:
                continue
            row: dict[str, float | int | str] = {
                "cell_id": cell_id,
                "cycle_index": int(cycle_index),
                "step_type": step_type,
            }
            row.update(curve)
            rows.append(row)
        return FeatureTable(self.name, pd.DataFrame(rows))

    def _curve_features(self, segment: pd.DataFrame) -> dict[str, float] | None:
        voltage = segment["voltage_v"].to_numpy(dtype=float)
        if segment["step_type"].iloc[0] == "charge" and "charge_capacity_ah" in segment:
            q = _capacity_vector(segment, "charge_capacity_ah")
        elif segment["step_type"].iloc[0] == "discharge" and "discharge_capacity_ah" in segment:
            q = _capacity_vector(segment, "discharge_capacity_ah")
        else:
            q = cumulative_capacity_ah(segment)

        v_grid, q_by_v = resample_xy(
            voltage, q, points=self.config.voltage_grid_points, min_points=self.config.min_points_for_curve
        )
        q_grid, v_by_q = resample_xy(
            q, voltage, points=self.config.capacity_grid_points, min_points=self.config.min_points_for_curve
        )
        if len(v_grid) < self.config.min_points_for_curve or len(q_grid) < self.config.min_points_for_curve:
            return None

        q_smooth = safe_savgol(q_by_v, self.config.smoothing_window, self.config.smoothing_polyorder)
        v_smooth = safe_savgol(v_by_q, self.config.smoothing_window, self.config.smoothing_polyorder)
        dqdv = safe_gradient(q_smooth, v_grid)
        dvdq = safe_gradient(v_smooth, q_grid)
        if len(dqdv) == 0 or len(dvdq) == 0:
            return None

        features: dict[str, float] = {}
        features.update(describe_array("ica_dqdv", dqdv))
        features.update(describe_array("dva_dvdq", dvdq))
        features["ica_area"] = trapezoid(np.nan_to_num(dqdv, nan=0.0), v_grid)
        features["dva_area"] = trapezoid(np.nan_to_num(dvdq, nan=0.0), q_grid)
        features.update(
            _peak_features(
                "ica",
                v_grid,
                dqdv,
                self.config.max_peaks,
                self.config.peak_prominence_noise_multiplier,
            )
        )
        features.update(
            _peak_features(
                "dva",
                q_grid,
                np.abs(dvdq),
                self.config.max_peaks,
                self.config.peak_prominence_noise_multiplier,
            )
        )
        return features


def _capacity_vector(segment: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(segment[column], errors="coerce")
    if values.notna().sum() < 5:
        return cumulative_capacity_ah(segment)
    arr = values.interpolate(limit_direction="both").to_numpy(dtype=float)
    return arr - np.nanmin(arr)


def _peak_features(
    prefix: str,
    x: np.ndarray,
    y: np.ndarray,
    max_peaks: int,
    prominence_noise_multiplier: float,
) -> dict[str, float]:
    result: dict[str, float] = {}
    clean_y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if len(clean_y) < 5:
        return result
    prominence = _mad_noise_prominence(clean_y, prominence_noise_multiplier)
    peaks, properties = find_peaks(clean_y, prominence=prominence)
    if len(peaks) == 0:
        result[f"{prefix}_peak_count"] = 0
        return result
    prominences = properties.get("prominences", np.zeros(len(peaks)))
    order = np.argsort(prominences)[::-1][:max_peaks]
    selected = peaks[order]
    widths = peak_widths(clean_y, selected, rel_height=0.5)[0]
    dx = float(np.nanmedian(np.diff(x))) if len(x) > 1 else 1.0
    result[f"{prefix}_peak_count"] = int(len(peaks))
    for rank, peak_index in enumerate(selected, start=1):
        result[f"{prefix}_peak_{rank}_x"] = float(x[peak_index])
        result[f"{prefix}_peak_{rank}_height"] = float(clean_y[peak_index])
        result[f"{prefix}_peak_{rank}_prominence"] = float(prominences[order[rank - 1]])
        result[f"{prefix}_peak_{rank}_width_x"] = float(widths[rank - 1] * dx)
    return result


def _mad_noise_prominence(values: np.ndarray, multiplier: float) -> float:
    """Estimate a peak prominence floor from MAD of first differences."""

    diffs = np.diff(values)
    if len(diffs) == 0:
        return 1e-12
    median = float(np.nanmedian(diffs))
    mad = float(np.nanmedian(np.abs(diffs - median)))
    noise_sigma = mad / 0.6745 if mad > 0 else float(np.nanstd(diffs))
    if not np.isfinite(noise_sigma) or noise_sigma <= 0:
        amplitude = float(np.nanmax(values) - np.nanmin(values))
        return max(amplitude * 1e-6, 1e-12)
    return max(noise_sigma * multiplier, 1e-12)
