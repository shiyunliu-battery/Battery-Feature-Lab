"""Conservative observation and stress flags without mechanism attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from battery_feature_lab.schemas import DiagnosticConfig


def build_degradation_tags(
    cycle_features: pd.DataFrame | None = None,
    ica_dva_features: pd.DataFrame | None = None,
    stress_features: pd.DataFrame | None = None,
    relaxation_features: pd.DataFrame | None = None,
    config: DiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Build observation-level signals only; no LLI/LAM mechanism labels are emitted."""

    del ica_dva_features, relaxation_features
    config = config or DiagnosticConfig()
    cycle_features = _empty_if_none(cycle_features)
    stress_features = _empty_if_none(stress_features)
    records: list[dict[str, object]] = []
    cells = set()
    for table in (cycle_features, stress_features):
        if "cell_id" in table.columns:
            cells.update(table["cell_id"].dropna().astype(str).unique())
    for cell_id in sorted(cells):
        cell_cycle = cycle_features[cycle_features.get("cell_id", pd.Series(dtype=str)) == cell_id]
        cell_stress = stress_features[stress_features.get("cell_id", pd.Series(dtype=str)) == cell_id]
        records.extend(_capacity_trend_observations(cell_id, cell_cycle, config))
        records.extend(_stress_observations(cell_id, cell_stress, stress_features, config))
    return pd.DataFrame(records)


def _capacity_trend_observations(cell_id, cycles, config):
    if cycles.empty or "discharge_capacity_ah" not in cycles.columns:
        return []
    ordered = cycles.sort_values("cycle_index")
    capacity = pd.to_numeric(ordered["discharge_capacity_ah"], errors="coerce").to_numpy(dtype=float)
    cycle = pd.to_numeric(ordered["cycle_index"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(capacity) & np.isfinite(cycle)
    if valid.sum() < config.min_trend_points:
        return []
    first_capacity = capacity[valid][0]
    if not np.isfinite(first_capacity) or abs(first_capacity) <= 1e-12:
        return []
    trend = mann_kendall_sen_slope(cycle[valid], capacity[valid] / first_capacity)
    if trend["sen_slope"] < 0 and trend["p_value"] < config.trend_p_value_alpha:
        return [_observation(
            cell_id, "capacity_decreasing_trend",
            f"Mann-Kendall trend test detects monotonic discharge-capacity decrease: p={trend['p_value']:.3g}, Sen slope={trend['sen_slope']:.3e} per cycle.",
            cycle_start=int(np.nanmin(cycle[valid])), cycle_end=int(np.nanmax(cycle[valid])),
            p_value=trend["p_value"], sen_slope=trend["sen_slope"],
        )]
    return []


def _stress_observations(cell_id, stress, all_stress, config):
    if stress.empty:
        return []
    row = stress.iloc[0]
    observations = []
    high_soc_rest = float(row.get("high_soc_rest_fraction", np.nan))
    if str(row.get("soc_source", "")).lower() == "provided" and np.isfinite(high_soc_rest):
        triggered, basis = _evaluate_stress_metric(high_soc_rest, _column_values(all_stress, "high_soc_rest_fraction"), config.high_soc_rest_fraction_threshold, "percentile", config.stress_percentile)
        if triggered:
            observations.append(_observation(cell_id, "high_soc_rest_exposure", f"High-SOC rest fraction {high_soc_rest:.3g} flagged by {basis}."))
    c_rate_variance = float(row.get("c_rate_variance", np.nan))
    if np.isfinite(c_rate_variance):
        triggered, basis = _evaluate_stress_metric(c_rate_variance, _column_values(all_stress, "c_rate_variance"), config.c_rate_variance_threshold, "mad", config.stress_mad_threshold)
        if triggered:
            observations.append(_observation(cell_id, "dynamic_current_variance", f"C-rate variance {c_rate_variance:.3g} flagged by {basis}."))
    max_discharge_c = float(row.get("max_instant_discharge_c_rate", np.nan))
    if config.datasheet_max_discharge_c_rate is not None and np.isfinite(max_discharge_c) and max_discharge_c > config.datasheet_max_discharge_c_rate * config.max_discharge_c_rate_fraction:
        observations.append(_observation(cell_id, "high_instantaneous_discharge_rate", f"Maximum instantaneous discharge C-rate {max_discharge_c:.2f} exceeds {config.max_discharge_c_rate_fraction:.0%} of datasheet limit {config.datasheet_max_discharge_c_rate:.2f}."))
    return observations


def mann_kendall_sen_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y); x = x[valid]; y = y[valid]
    if len(x) < 2:
        return {"p_value": float("nan"), "sen_slope": float("nan"), "kendall_tau": float("nan")}
    tau, p_value = kendalltau(x, y)
    slopes = []
    for i in range(len(x)-1):
        dx = x[i+1:] - x[i]; dy = y[i+1:] - y[i]; valid_dx = np.abs(dx) > 1e-12
        slopes.extend((dy[valid_dx] / dx[valid_dx]).tolist())
    return {"p_value": float(p_value) if np.isfinite(p_value) else float("nan"), "sen_slope": float(np.median(slopes)) if slopes else float("nan"), "kendall_tau": float(tau) if np.isfinite(tau) else float("nan")}


def _evaluate_stress_metric(value, population, absolute_threshold, batch_method, batch_param):
    if absolute_threshold is not None and value > absolute_threshold:
        return True, f"absolute threshold {absolute_threshold:.3g} (config/domain knowledge)"
    threshold = _batch_threshold(population, batch_method, batch_param)
    if np.isfinite(threshold) and value > threshold:
        label = f"batch P{batch_param * 100:.0f}" if batch_method == "percentile" else f"batch median+{batch_param:g}*MAD"
        return True, f"batch-relative outlier (> {label} = {threshold:.3g})"
    return False, ""


def _batch_threshold(population, method, param):
    values = np.asarray(population, dtype=float); values = values[np.isfinite(values)]
    if method == "percentile":
        return float(np.quantile(values, param)) if len(values) >= 2 else float("nan")
    if method == "mad":
        if len(values) < 3:
            return float("nan")
        median = float(np.median(values)); mad = float(np.median(np.abs(values-median))); robust_sigma = 1.4826 * mad
        return float(median + param * robust_sigma) if np.isfinite(robust_sigma) and robust_sigma > 1e-12 else float("nan")
    return float("nan")


def _column_values(frame, column):
    return pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)


def _observation(cell_id, signal, evidence, *, cycle_start=None, cycle_end=None, p_value=None, sen_slope=None):
    return {"cell_id": cell_id, "signal": signal, "kind": "observation", "evidence": evidence, "cycle_start": cycle_start, "cycle_end": cycle_end, "p_value": p_value, "sen_slope": sen_slope}


def _empty_if_none(frame):
    return frame if frame is not None else pd.DataFrame()
