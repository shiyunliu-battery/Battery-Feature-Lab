"""Curve resampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def monotonic_average(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by x and average duplicate x values."""

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.array([]), np.array([])
    data = pd.DataFrame({"x": x[mask], "y": y[mask]}).sort_values("x")
    grouped = data.groupby("x", as_index=False)["y"].mean()
    return grouped["x"].to_numpy(), grouped["y"].to_numpy()


def resample_xy(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray | None = None,
    points: int = 1000,
    min_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample y(x) onto a stable grid."""

    x_clean, y_clean = monotonic_average(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if len(x_clean) < min_points:
        return np.array([]), np.array([])
    if grid is None:
        x_min = float(np.nanmin(x_clean))
        x_max = float(np.nanmax(x_clean))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            return np.array([]), np.array([])
        grid = np.linspace(x_min, x_max, points)
    else:
        grid = np.asarray(grid, dtype=float)
        grid = grid[(grid >= x_clean.min()) & (grid <= x_clean.max())]
        if len(grid) < min_points:
            return np.array([]), np.array([])
    return grid, np.interp(grid, x_clean, y_clean)


def overlapping_grid(a: np.ndarray, b: np.ndarray, points: int) -> np.ndarray:
    """Create a grid over the overlap of two monotonic arrays."""

    if len(a) == 0 or len(b) == 0:
        return np.array([])
    lo = max(float(np.nanmin(a)), float(np.nanmin(b)))
    hi = min(float(np.nanmax(a)), float(np.nanmax(b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([])
    return np.linspace(lo, hi, points)
