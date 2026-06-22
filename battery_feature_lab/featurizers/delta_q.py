"""Early-life Delta Q(V) feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_feature_lab.core.integration import cumulative_capacity_ah, trapezoid
from battery_feature_lab.core.resampling import overlapping_grid, resample_xy
from battery_feature_lab.featurizers.base import BaseFeaturizer
from battery_feature_lab.featurizers.common import describe_array
from battery_feature_lab.schemas import FeatureTable


class DeltaQFeaturizer(BaseFeaturizer):
    """Extract statistics from Q_target(V) - Q_reference(V) discharge curves."""

    name = "delta_q_features"

    def extract(self, frame: pd.DataFrame) -> FeatureTable:
        rows: list[dict[str, float | int | str]] = []
        ref_cycle = self.config.early_reference_cycle
        target_cycle = self.config.early_target_cycle
        for cell_id, cell in frame.groupby("cell_id", sort=True):
            ref = _discharge_qv(
                cell[cell["cycle_index"] == ref_cycle],
                min_points=self.config.min_points_for_curve,
                grid_points=self.config.delta_q_voltage_points,
            )
            target = _discharge_qv(
                cell[cell["cycle_index"] == target_cycle],
                min_points=self.config.min_points_for_curve,
                grid_points=self.config.delta_q_voltage_points,
            )
            if ref is None or target is None:
                continue
            ref_v, ref_q = ref
            target_v, target_q = target
            grid = overlapping_grid(ref_v, target_v, self.config.delta_q_voltage_points)
            if len(grid) < self.config.min_points_for_curve:
                continue
            _, ref_q_interp = resample_xy(ref_v, ref_q, grid=grid)
            _, target_q_interp = resample_xy(target_v, target_q, grid=grid)
            if len(ref_q_interp) != len(target_q_interp) or len(ref_q_interp) == 0:
                continue
            delta_q = target_q_interp - ref_q_interp
            row: dict[str, float | int | str] = {
                "cell_id": cell_id,
                "reference_cycle": ref_cycle,
                "target_cycle": target_cycle,
                "voltage_min_v": float(grid.min()),
                "voltage_max_v": float(grid.max()),
                "delta_q_area_ah_v": trapezoid(delta_q, grid),
                "delta_q_abs_area_ah_v": trapezoid(np.abs(delta_q), grid),
                "delta_q_l1": float(np.sum(np.abs(delta_q))),
                "delta_q_l2": float(np.sqrt(np.sum(delta_q**2))),
                "delta_q_variance": float(np.var(delta_q, ddof=1)),
                "delta_q_log_variance": float(np.log10(np.var(delta_q, ddof=1) + 1e-18)),
                "delta_q_min_voltage_v": float(grid[int(np.nanargmin(delta_q))]),
                "delta_q_max_voltage_v": float(grid[int(np.nanargmax(delta_q))]),
            }
            row.update(describe_array("delta_q_ah", delta_q))
            row.update(_window_stats(grid, delta_q))
            rows.append(row)
        return FeatureTable(self.name, pd.DataFrame(rows))


def _discharge_qv(
    cycle: pd.DataFrame, *, min_points: int, grid_points: int
) -> tuple[np.ndarray, np.ndarray] | None:
    discharge = cycle[cycle["step_type"] == "discharge"].copy()
    if len(discharge) < min_points:
        return None
    voltage = discharge["voltage_v"].to_numpy(dtype=float)
    if (
        "discharge_capacity_ah" in discharge.columns
        and discharge["discharge_capacity_ah"].notna().sum() >= min_points
    ):
        q = discharge["discharge_capacity_ah"].to_numpy(dtype=float)
        q = q - np.nanmin(q)
    else:
        q = cumulative_capacity_ah(discharge)
    grid, q_interp = resample_xy(voltage, q, points=grid_points, min_points=min_points)
    if len(grid) < min_points:
        return None
    return grid, q_interp


def _window_stats(voltage: np.ndarray, delta_q: np.ndarray, windows: int = 5) -> dict[str, float]:
    result: dict[str, float] = {}
    edges = np.linspace(float(voltage.min()), float(voltage.max()), windows + 1)
    for idx in range(windows):
        mask = (voltage >= edges[idx]) & (voltage <= edges[idx + 1])
        values = delta_q[mask]
        if len(values) == 0:
            continue
        prefix = f"delta_q_window_{idx + 1}"
        result[f"{prefix}_mean"] = float(np.nanmean(values))
        result[f"{prefix}_variance"] = float(np.nanvar(values, ddof=1)) if len(values) > 1 else 0.0
        result[f"{prefix}_min"] = float(np.nanmin(values))
        result[f"{prefix}_max"] = float(np.nanmax(values))
    return result
