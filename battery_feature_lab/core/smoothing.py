"""Smoothing and numerical differentiation helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def safe_savgol(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Apply Savitzky-Golay smoothing with safe fallbacks for short arrays."""

    values = np.asarray(values, dtype=float)
    if len(values) < 5 or window < 5:
        return values
    adjusted = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if adjusted <= polyorder:
        adjusted = polyorder + 2
        if adjusted % 2 == 0:
            adjusted += 1
    if adjusted > len(values) or adjusted < 5:
        return values
    return savgol_filter(values, adjusted, polyorder, mode="interp")


def safe_gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute dy/dx while suppressing divide-by-zero artifacts."""

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) < 3 or len(x) < 3:
        return np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = np.gradient(y, x)
    grad[~np.isfinite(grad)] = np.nan
    return grad
