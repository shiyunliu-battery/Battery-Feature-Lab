"""Rule-based degradation evidence tagging."""

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
    """Build interpretable degradation tags from extracted feature tables."""

    config = config or DiagnosticConfig()
    records: list[dict[str, object]] = []
    cycle_features = _empty_if_none(cycle_features)
    ica_dva_features = _empty_if_none(ica_dva_features)
    stress_features = _empty_if_none(stress_features)
    relaxation_features = _empty_if_none(relaxation_features)

    cells = set()
    for table in (cycle_features, ica_dva_features, stress_features, relaxation_features):
        if "cell_id" in table.columns:
            cells.update(table["cell_id"].dropna().astype(str).unique())

    for cell_id in sorted(cells):
        cell_cycle = cycle_features[cycle_features.get("cell_id", pd.Series(dtype=str)) == cell_id]
        cell_ica = ica_dva_features[ica_dva_features.get("cell_id", pd.Series(dtype=str)) == cell_id]
        cell_stress = stress_features[stress_features.get("cell_id", pd.Series(dtype=str)) == cell_id]
        cell_relax = relaxation_features[
            relaxation_features.get("cell_id", pd.Series(dtype=str)) == cell_id
        ]
        records.extend(_capacity_fade_tags(cell_id, cell_cycle, config))
        records.extend(_ica_shift_tags(cell_id, cell_ica, config))
        records.extend(_stress_tags(cell_id, cell_stress, stress_features, config))
        records.extend(_relaxation_tags(cell_id, cell_relax, config))

    return pd.DataFrame(records)


def _capacity_fade_tags(
    cell_id: str, cycles: pd.DataFrame, config: DiagnosticConfig
) -> list[dict[str, object]]:
    if cycles.empty or "discharge_capacity_ah" not in cycles:
        return []
    ordered = cycles.sort_values("cycle_index")
    capacity = ordered["discharge_capacity_ah"].to_numpy(dtype=float)
    cycle = ordered["cycle_index"].to_numpy(dtype=float)
    valid = np.isfinite(capacity) & np.isfinite(cycle)
    if valid.sum() < config.min_trend_points:
        return []
    normalized = capacity[valid] / capacity[valid][0] if capacity[valid][0] else capacity[valid]
    trend = mann_kendall_sen_slope(cycle[valid], normalized)
    if trend["sen_slope"] < 0 and trend["p_value"] < config.trend_p_value_alpha:
        return [
            _tag(
                cell_id,
                "capacity_fade_detected",
                ["LLI", "LAM_PE", "LAM_NE"],
                "medium",
                "Mann-Kendall trend test detects monotonic capacity fade: "
                f"p={trend['p_value']:.3g}, Sen slope={trend['sen_slope']:.3e} per cycle.",
            )
        ]
    return []


def _ica_shift_tags(
    cell_id: str, ica: pd.DataFrame, config: DiagnosticConfig
) -> list[dict[str, object]]:
    if ica.empty or "ica_peak_1_x" not in ica.columns:
        return []
    discharge = ica[ica.get("step_type", "") == "discharge"].sort_values("cycle_index")
    if len(discharge) < config.min_trend_points:
        return []
    peak_x = discharge["ica_peak_1_x"].to_numpy(dtype=float)
    cycle = discharge["cycle_index"].to_numpy(dtype=float)
    valid = np.isfinite(peak_x) & np.isfinite(cycle)
    if valid.sum() < config.min_trend_points:
        return []
    trend = mann_kendall_sen_slope(cycle[valid], peak_x[valid])
    if trend["p_value"] >= config.trend_p_value_alpha or not np.isfinite(trend["sen_slope"]):
        return []
    mode = "LLI" if trend["sen_slope"] < 0 else "LAM_PE"
    return [
        _tag(
            cell_id,
            "ica_primary_peak_shift",
            [mode],
            "medium",
            "Mann-Kendall trend test detects primary ICA peak drift: "
            f"p={trend['p_value']:.3g}, Sen slope={trend['sen_slope']:.3e} V per cycle.",
        )
    ]


def _stress_tags(
    cell_id: str,
    stress: pd.DataFrame,
    all_stress: pd.DataFrame,
    config: DiagnosticConfig,
) -> list[dict[str, object]]:
    if stress.empty:
        return []
    row = stress.iloc[0]
    tags = []

    high_soc_rest = float(row.get("high_soc_rest_fraction", np.nan))
    if np.isfinite(high_soc_rest):
        triggered, basis = _evaluate_stress_metric(
            value=high_soc_rest,
            population=_column_values(all_stress, "high_soc_rest_fraction"),
            absolute_threshold=config.high_soc_rest_fraction_threshold,
            batch_method="percentile",
            batch_param=config.stress_percentile,
        )
        if triggered:
            tags.append(
                _tag(
                    cell_id,
                    "high_soc_rest_exposure",
                    ["LAM_PE", "LLI"],
                    "medium",
                    f"High-SOC rest fraction {high_soc_rest:.3g} flagged by {basis}.",
                )
            )

    c_rate_variance = float(row.get("c_rate_variance", np.nan))
    if np.isfinite(c_rate_variance):
        triggered, basis = _evaluate_stress_metric(
            value=c_rate_variance,
            population=_column_values(all_stress, "c_rate_variance"),
            absolute_threshold=config.c_rate_variance_threshold,
            batch_method="mad",
            batch_param=config.stress_mad_threshold,
        )
        if triggered:
            tags.append(
                _tag(
                    cell_id,
                    "dynamic_current_variance",
                    ["LAM_NE", "resistance_growth"],
                    "low",
                    f"Normalized C-rate variance {c_rate_variance:.3g} flagged by {basis}.",
                )
            )

    max_discharge_c = float(row.get("max_instant_discharge_c_rate", np.nan))
    if (
        config.datasheet_max_discharge_c_rate is not None
        and np.isfinite(max_discharge_c)
        and max_discharge_c
        > config.datasheet_max_discharge_c_rate * config.max_discharge_c_rate_fraction
    ):
        tags.append(
            _tag(
                cell_id,
                "high_instantaneous_discharge_rate",
                ["LAM_NE", "resistance_growth"],
                "medium",
                f"Maximum instantaneous discharge C-rate {max_discharge_c:.2f} exceeds "
                f"{config.max_discharge_c_rate_fraction:.0%} of datasheet limit "
                f"{config.datasheet_max_discharge_c_rate:.2f}.",
            )
        )
    return tags


def _relaxation_tags(
    cell_id: str, relaxation: pd.DataFrame, config: DiagnosticConfig
) -> list[dict[str, object]]:
    if relaxation.empty or "rest_exp_tau_s" not in relaxation.columns:
        return []
    ordered = relaxation.sort_values("cycle_index")
    tau = ordered["rest_exp_tau_s"].to_numpy(dtype=float)
    cycle = ordered["cycle_index"].to_numpy(dtype=float)
    valid = np.isfinite(tau) & np.isfinite(cycle)
    if valid.sum() < config.min_trend_points:
        return []
    trend = mann_kendall_sen_slope(cycle[valid], tau[valid])
    if trend["sen_slope"] > 0 and trend["p_value"] < config.trend_p_value_alpha:
        return [
            _tag(
                cell_id,
                "relaxation_tau_increase",
                ["resistance_growth", "transport_limitation"],
                "low",
                "Mann-Kendall trend test detects increasing relaxation time constant: "
                f"p={trend['p_value']:.3g}, Sen slope={trend['sen_slope']:.3e} s per cycle.",
            )
        ]
    return []


def mann_kendall_sen_slope(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return the Mann-Kendall trend p-value and Sen's slope for a monotonic trend.

    The Mann-Kendall test statistic is Kendall's tau between the series and its (time/cycle)
    index, so ``scipy.stats.kendalltau`` computes the test directly (exact for small n, asymptotic
    with tie handling otherwise). Sen's slope is the Theil-Sen median of pairwise slopes, the trend
    magnitude estimator conventionally paired with Mann-Kendall. Power is low for very short
    series; callers should require a minimum sample size (see ``DiagnosticConfig.min_trend_points``).
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return {"p_value": float("nan"), "sen_slope": float("nan"), "kendall_tau": float("nan")}
    tau, p_value = kendalltau(x, y)
    slopes: list[float] = []
    for i in range(len(x) - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        valid_dx = np.abs(dx) > 1e-12
        slopes.extend((dy[valid_dx] / dx[valid_dx]).tolist())
    sen_slope = float(np.median(slopes)) if slopes else float("nan")
    return {
        "p_value": float(p_value) if np.isfinite(p_value) else float("nan"),
        "sen_slope": sen_slope,
        "kendall_tau": float(tau) if np.isfinite(tau) else float("nan"),
    }


def _evaluate_stress_metric(
    value: float,
    population: np.ndarray,
    absolute_threshold: float | None,
    batch_method: str,
    batch_param: float,
) -> tuple[bool, str]:
    """Dual-criterion stress evaluation.

    The absolute threshold encodes domain knowledge (cell spec / application limit) and is
    single-cell capable. The batch criterion is a *supplementary* outlier signal that requires
    a population of cells and is reported as relative, never as an absolute physical judgment.
    Returns ``(triggered, basis)`` where ``basis`` names which criterion fired.
    """

    if absolute_threshold is not None and value > absolute_threshold:
        return True, f"absolute threshold {absolute_threshold:.3g} (config/domain knowledge)"
    batch_threshold = _batch_threshold(population, batch_method, batch_param)
    if np.isfinite(batch_threshold) and value > batch_threshold:
        if batch_method == "percentile":
            label = f"batch P{batch_param * 100:.0f}"
        else:
            label = f"batch median+{batch_param:g}*MAD"
        return True, f"batch-relative outlier (> {label} = {batch_threshold:.3g})"
    return False, ""


def _batch_threshold(population: np.ndarray, method: str, param: float) -> float:
    values = np.asarray(population, dtype=float)
    values = values[np.isfinite(values)]
    if method == "percentile":
        if len(values) < 2:
            return float("nan")
        return float(np.quantile(values, param))
    if method == "mad":
        # Robust outlier threshold: median + param * (1.4826 * MAD). The 1.4826 factor scales the
        # median absolute deviation to a standard-deviation-consistent estimator under normality,
        # so ``param`` is interpretable as a number of (robust) sigmas. Unlike mean+k*std this does
        # not let a single extreme cell inflate the spread and mask itself.
        if len(values) < 3:
            return float("nan")
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma <= 1e-12 or not np.isfinite(robust_sigma):
            return float("nan")
        return float(median + param * robust_sigma)
    return float("nan")


def _column_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    series = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")
    return series.to_numpy(dtype=float)


def _tag(
    cell_id: str,
    signal: str,
    possible_modes: list[str],
    confidence: str,
    evidence: str,
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "signal": signal,
        "possible_modes": possible_modes,
        "confidence": confidence,
        "evidence": evidence,
    }


def _empty_if_none(frame: pd.DataFrame | None) -> pd.DataFrame:
    return frame if frame is not None else pd.DataFrame()
