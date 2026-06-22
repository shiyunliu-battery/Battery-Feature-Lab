"""Optional SHAP analysis adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd


def shap_feature_importance(model, features: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    """Return mean absolute SHAP values for a fitted tree/model object.

    This function imports SHAP lazily so the core pipeline remains lightweight.
    """

    try:
        import shap  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install battery-feature-lab[explain] to use SHAP reporting.") from exc

    numeric = features.select_dtypes(include=[np.number]).copy()
    if len(numeric) > max_rows:
        numeric = numeric.sample(max_rows, random_state=42)
    explainer = shap.Explainer(model, numeric)
    values = explainer(numeric)
    arr = values.values
    if arr.ndim == 3:
        arr = np.mean(np.abs(arr), axis=2)
    importance = np.mean(np.abs(arr), axis=0)
    return pd.DataFrame({"feature": numeric.columns, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False
    )
