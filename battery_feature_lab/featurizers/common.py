"""Shared featurizer utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def describe_array(prefix: str, values: np.ndarray) -> dict[str, float]:
    """Return robust statistics for a numeric vector."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q05": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"{prefix}_q95": float("nan"),
            f"{prefix}_skew": float("nan"),
            f"{prefix}_kurtosis": float("nan"),
        }
    nearly_constant = float(np.std(values)) < 1e-12
    return {
        f"{prefix}_count": int(len(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_skew": float(skew(values, bias=False)) if len(values) > 2 and not nearly_constant else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(values, fisher=True, bias=False))
        if len(values) > 3 and not nearly_constant
        else 0.0,
    }


def duration_s(frame: pd.DataFrame) -> float:
    """Return elapsed duration in seconds for a frame."""

    if frame.empty or "time_s" not in frame:
        return 0.0
    values = pd.to_numeric(frame["time_s"], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(values.max() - values.min())


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator with NaN on invalid denominator."""

    if denominator is None or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)
