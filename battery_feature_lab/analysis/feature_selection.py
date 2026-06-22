"""Feature selection and statistical screening utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def screen_features(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    min_non_null_fraction: float = 0.8,
    max_abs_correlation: float = 0.98,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Rank numeric features using correlation and mutual information."""

    numeric = features.select_dtypes(include=[np.number]).copy()
    target = pd.to_numeric(target, errors="coerce")
    aligned = numeric.join(target.rename("__target__")).dropna(subset=["__target__"])
    rows: list[dict[str, float | str]] = []
    for column in numeric.columns:
        series = aligned[column]
        non_null_fraction = float(series.notna().mean()) if len(series) else 0.0
        if non_null_fraction < min_non_null_fraction:
            continue
        pair = aligned[[column, "__target__"]].dropna()
        if len(pair) < 5 or pair[column].nunique() < 2:
            continue
        pearson = float(pair[column].corr(pair["__target__"], method="pearson"))
        spearman = float(pair[column].corr(pair["__target__"], method="spearman"))
        rows.append(
            {
                "feature": column,
                "non_null_fraction": non_null_fraction,
                "pearson": pearson,
                "spearman": spearman,
                "abs_pearson": abs(pearson) if np.isfinite(pearson) else 0.0,
                "abs_spearman": abs(spearman) if np.isfinite(spearman) else 0.0,
            }
        )
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    mi_values = _mutual_information(numeric[ranked["feature"].tolist()], target)
    ranked["mutual_information"] = ranked["feature"].map(mi_values)
    ranked = ranked.sort_values(
        ["mutual_information", "abs_spearman", "abs_pearson"], ascending=False
    ).reset_index(drop=True)
    ranked["selected"] = _select_non_redundant(
        numeric, ranked["feature"].tolist(), max_abs_correlation=max_abs_correlation, top_k=top_k
    )
    return ranked


def variance_inflation_factors(features: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF-like multicollinearity scores with linear least squares."""

    numeric = features.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    rows = []
    for column in numeric.columns:
        y = numeric[column].to_numpy(dtype=float)
        x = numeric.drop(columns=[column]).to_numpy(dtype=float)
        if x.shape[1] == 0 or np.nanstd(y) < 1e-12:
            vif = float("nan")
        else:
            x = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            pred = x @ beta
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = float(1.0 / max(1.0 - r2, 1e-12))
        rows.append({"feature": column, "vif": vif})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def _mutual_information(features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    joined = features.join(target.rename("__target__")).dropna()
    if len(joined) < 5:
        return {column: float("nan") for column in features.columns}
    x = joined[features.columns].to_numpy(dtype=float)
    y = joined["__target__"].to_numpy(dtype=float)
    values = mutual_info_regression(x, y, random_state=42)
    return {column: float(value) for column, value in zip(features.columns, values)}


def _select_non_redundant(
    numeric: pd.DataFrame,
    ranked_features: list[str],
    *,
    max_abs_correlation: float,
    top_k: int | None,
) -> list[bool]:
    selected: list[str] = []
    decisions: list[bool] = []
    corr = numeric[ranked_features].corr().abs()
    for feature in ranked_features:
        redundant = any(corr.loc[feature, existing] > max_abs_correlation for existing in selected)
        keep = not redundant and (top_k is None or len(selected) < top_k)
        decisions.append(keep)
        if keep:
            selected.append(feature)
    return decisions
