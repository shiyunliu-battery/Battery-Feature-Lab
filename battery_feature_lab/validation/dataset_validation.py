"""Empirical validation of early-life features against observed cycle life.

The validation criterion follows Severson et al., Nature Energy 2019: features extracted from
early cycles (notably the variance of Delta Q(V)) should correlate strongly with the eventual
cycle life. Reproducing that correlation on a real dataset is direct evidence that the feature
extraction is implemented correctly, rather than merely "looking reasonable".

This module is dataset-agnostic: it operates on the normalized time-series frame produced by the
BDS reader, so it works with the bundled synthetic populations, MATR/Severson exports converted to
the canonical schema, or any folder of per-cell cycler CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from battery_feature_lab.featurizers.cycle_summary import CycleSummaryFeaturizer
from battery_feature_lab.featurizers.delta_q import DeltaQFeaturizer
from battery_feature_lab.schemas import FeatureConfig


@dataclass(frozen=True)
class CycleLifeResult:
    """Per-cell observed life with a censoring flag for cells that never reach EOL."""

    table: pd.DataFrame  # columns: cell_id, cycle_life, censored, initial_capacity_ah


def compute_cycle_life(
    cycle_features: pd.DataFrame,
    *,
    capacity_column: str = "discharge_capacity_ah",
    eol_fraction: float = 0.8,
    reference: str = "initial",
    initial_window: int = 5,
) -> CycleLifeResult:
    """Compute cycles-to-end-of-life per cell from a cycle-summary table.

    End of life is the first cycle whose capacity falls to ``eol_fraction`` of the reference
    capacity. ``reference="initial"`` uses the median of the first ``initial_window`` cycles
    (robust to a noisy first cycle); cells that never reach EOL are reported as right-censored
    with their last observed cycle, and excluded from correlation by the caller.
    """

    if cycle_features.empty or capacity_column not in cycle_features.columns:
        return CycleLifeResult(pd.DataFrame(columns=["cell_id", "cycle_life", "censored", "initial_capacity_ah"]))

    records: list[dict[str, object]] = []
    for cell_id, cell in cycle_features.groupby("cell_id", sort=True):
        ordered = cell.sort_values("cycle_index")
        capacity = pd.to_numeric(ordered[capacity_column], errors="coerce").to_numpy(dtype=float)
        cycle = pd.to_numeric(ordered["cycle_index"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(capacity) & np.isfinite(cycle)
        capacity, cycle = capacity[valid], cycle[valid]
        if len(capacity) < initial_window + 1:
            continue
        if reference == "initial":
            ref_cap = float(np.nanmedian(capacity[:initial_window]))
        else:
            ref_cap = float(np.nanmax(capacity))
        if not np.isfinite(ref_cap) or ref_cap <= 0:
            continue
        threshold = eol_fraction * ref_cap
        below = np.where(capacity <= threshold)[0]
        if len(below) > 0:
            life = float(cycle[below[0]])
            censored = False
        else:
            life = float(cycle[-1])
            censored = True
        records.append(
            {"cell_id": cell_id, "cycle_life": life, "censored": censored, "initial_capacity_ah": ref_cap}
        )
    return CycleLifeResult(pd.DataFrame(records))


def correlation_report(
    frame: pd.DataFrame, feature_columns: list[str], target_column: str
) -> pd.DataFrame:
    """Pearson r, Spearman rho and linear R^2 of each feature against the target.

    Spearman is the primary statistic (monotonic, outlier-robust — appropriate for battery data);
    Pearson and R^2 are reported on the (typically log-transformed) target for comparability with
    Severson et al. Rows are sorted by descending |Spearman|.
    """

    rows: list[dict[str, object]] = []
    target = pd.to_numeric(frame.get(target_column), errors="coerce").to_numpy(dtype=float)
    for column in feature_columns:
        if column not in frame.columns:
            continue
        feature = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(feature) & np.isfinite(target)
        n = int(mask.sum())
        if n < 3 or np.nanstd(feature[mask]) == 0:
            continue
        pearson_r = float(pearsonr(feature[mask], target[mask])[0])
        spearman_rho = float(spearmanr(feature[mask], target[mask])[0])
        rows.append(
            {
                "feature": column,
                "n": n,
                "spearman_rho": spearman_rho,
                "pearson_r": pearson_r,
                "r2": pearson_r**2,
            }
        )
    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.reindex(report["spearman_rho"].abs().sort_values(ascending=False).index)
        report = report.reset_index(drop=True)
    return report


def validate_early_life_features(
    normalized: pd.DataFrame,
    *,
    config: FeatureConfig | None = None,
    eol_fraction: float = 0.8,
    candidate_features: list[str] | None = None,
) -> dict[str, object]:
    """Run the early-life featurizers and correlate their outputs with observed cycle life.

    Returns a dict with the merged per-cell table, the correlation report, and the headline
    Severson-style statistic for ``delta_q_log_variance`` vs ``log10(cycle life)``.
    """

    config = config or FeatureConfig()
    cycle_features = CycleSummaryFeaturizer(config).extract(normalized).frame
    delta_q_features = DeltaQFeaturizer(config).extract(normalized).frame

    life = compute_cycle_life(cycle_features, eol_fraction=eol_fraction).table
    uncensored = life[~life["censored"]].copy()
    uncensored["log_cycle_life"] = np.log10(uncensored["cycle_life"].clip(lower=1.0))

    merged = uncensored.merge(delta_q_features, on="cell_id", how="inner")

    if candidate_features is None:
        candidate_features = [
            c
            for c in delta_q_features.columns
            if c not in {"cell_id", "reference_cycle", "target_cycle"}
            and pd.api.types.is_numeric_dtype(delta_q_features[c])
        ]

    report = correlation_report(merged, candidate_features, "log_cycle_life")

    headline = {}
    if "delta_q_log_variance" in merged.columns and len(merged) >= 3:
        x = pd.to_numeric(merged["delta_q_log_variance"], errors="coerce").to_numpy(dtype=float)
        y = merged["log_cycle_life"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 3 and np.nanstd(x[mask]) > 0:
            r = float(pearsonr(x[mask], y[mask])[0])
            headline = {
                "feature": "delta_q_log_variance",
                "target": "log10(cycle_life)",
                "n_cells": int(mask.sum()),
                "pearson_r": r,
                "spearman_rho": float(spearmanr(x[mask], y[mask])[0]),
                "r2": r**2,
            }

    return {
        "n_cells_total": int(len(life)),
        "n_cells_uncensored": int(len(uncensored)),
        "merged_table": merged,
        "correlation_report": report,
        "severson_headline": headline,
    }
