"""JSONL context export for cell-level review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SAMPLE_CYCLE_LIMIT = 12
_TREND_MIN_POINTS = 3
_CAPACITY_BALANCE_MIN = 0.8
_CAPACITY_BALANCE_MAX = 1.2
_CE_MIN = 0.95
_CE_MAX = 1.05


def build_llm_context_records(
    tables: dict[str, pd.DataFrame],
    degradation_tags: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build compact cell-level records for review and reporting."""

    metadata = metadata or {}
    cells = set()
    for frame in tables.values():
        if frame is not None and "cell_id" in frame.columns:
            cells.update(frame["cell_id"].dropna().astype(str).unique())
    if degradation_tags is not None and "cell_id" in degradation_tags.columns:
        cells.update(degradation_tags["cell_id"].dropna().astype(str).unique())

    records: list[dict[str, Any]] = []
    for cell_id in sorted(cells):
        cell_tables = _cell_tables(tables, cell_id)
        tag_records = _tag_records(degradation_tags, cell_id)
        analysis_config = _build_analysis_config(metadata.get("analysis_config"))
        cell_context = _build_cell_context(
            metadata.get("cell_context"),
            analysis_config,
            cell_tables,
        )
        record: dict[str, Any] = {
            "schema_version": "2.1",
            "cell_id": cell_id,
            "cell_context": cell_context,
            "analysis_config": analysis_config,
            "summary": {},
            "dataset_overview": {},
            "data_quality": {},
            "cycle_life_summary": {},
            "feature_highlights": {},
            "diagnostic_evidence": tag_records,
            "features": {},
            "review_notes": {},
            "provenance": {"source": "battery-feature-lab"},
        }

        record["dataset_overview"] = _build_dataset_overview(cell_tables)
        record["data_quality"] = _build_data_quality(
            cell_tables,
            tag_records,
            cell_context,
            analysis_config,
        )
        record["cycle_life_summary"] = _build_cycle_life_summary(
            cell_tables.get("cycle_features"),
            cell_context,
        )
        record["feature_highlights"] = _build_feature_highlights(cell_tables)
        record["summary"] = _build_summary(record)
        record["review_notes"] = _build_review_notes(record)

        for name, subset in cell_tables.items():
            if "cycle_index" in subset.columns:
                record["features"][name] = _summarize_table(subset)
            else:
                record["features"][name] = _clean_record(subset.iloc[0].to_dict())
        records.append(record)
    return records


def write_llm_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write cell context records as JSON Lines."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_clean_record(record), ensure_ascii=False) + "\n")
    return output


def _summarize_table(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.select_dtypes(include=[np.number])
    summary: dict[str, Any] = {"row_count": int(len(frame))}
    if "cycle_index" in frame.columns:
        summary["cycle_min"] = int(frame["cycle_index"].min())
        summary["cycle_max"] = int(frame["cycle_index"].max())
    for column in numeric.columns:
        if column == "cycle_index":
            continue
        values = numeric[column].dropna()
        if values.empty:
            continue
        summary[column] = {
            "first": _json_scalar(values.iloc[0]),
            "last": _json_scalar(values.iloc[-1]),
            "mean": _json_scalar(values.mean()),
            "min": _json_scalar(values.min()),
            "max": _json_scalar(values.max()),
        }
    return summary


def _cell_tables(tables: dict[str, pd.DataFrame], cell_id: str) -> dict[str, pd.DataFrame]:
    cell_tables: dict[str, pd.DataFrame] = {}
    for name, frame in tables.items():
        if frame is None or frame.empty or "cell_id" not in frame.columns:
            continue
        subset = frame[frame["cell_id"].astype(str) == cell_id].copy()
        if not subset.empty:
            if "cycle_index" in subset.columns:
                subset = subset.sort_values("cycle_index")
            cell_tables[name] = subset
    return cell_tables


def _tag_records(degradation_tags: pd.DataFrame | None, cell_id: str) -> list[dict[str, Any]]:
    if degradation_tags is None or degradation_tags.empty or "cell_id" not in degradation_tags.columns:
        return []
    tags = degradation_tags[degradation_tags["cell_id"].astype(str) == cell_id]
    records = [_clean_record(row) for row in tags.to_dict("records")]
    for record in records:
        signal = str(record.get("signal", ""))
        record["interpretation_hint"] = _signal_interpretation(signal)
    return records


def _build_analysis_config(raw: Any) -> dict[str, Any]:
    config = _clean_record(raw or {})
    reader = config.get("reader_config", {}) if isinstance(config, dict) else {}
    features = config.get("feature_config", {}) if isinstance(config, dict) else {}
    diagnostics = config.get("diagnostic_config", {}) if isinstance(config, dict) else {}
    return {
        "reader_config": reader,
        "feature_config": features,
        "diagnostic_config": diagnostics,
        "key_parameters": {
            "reference_cycle": features.get("early_reference_cycle"),
            "target_cycle": features.get("early_target_cycle"),
            "nominal_capacity_ah": features.get("nominal_capacity_ah"),
            "high_soc_level": features.get("high_soc_level"),
            "histogram_bins": features.get("histogram_bins"),
            "voltage_grid_points": features.get("voltage_grid_points"),
            "capacity_grid_points": features.get("capacity_grid_points"),
            "delta_q_voltage_points": features.get("delta_q_voltage_points"),
            "trend_p_value_alpha": diagnostics.get("trend_p_value_alpha"),
            "min_trend_points": diagnostics.get("min_trend_points"),
            "high_soc_rest_fraction_threshold": diagnostics.get(
                "high_soc_rest_fraction_threshold"
            ),
            "c_rate_variance_threshold": diagnostics.get("c_rate_variance_threshold"),
            "datasheet_max_discharge_c_rate": diagnostics.get(
                "datasheet_max_discharge_c_rate"
            ),
        },
    }


def _build_cell_context(
    raw: Any,
    analysis_config: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    context = _clean_record(raw or {})
    reader = analysis_config.get("reader_config", {})
    features = analysis_config.get("feature_config", {})
    normalized = tables.get("normalized_timeseries")
    cycle_features = tables.get("cycle_features")

    nominal_capacity = context.get("nominal_capacity_ah")
    if nominal_capacity is None:
        nominal_capacity = features.get("nominal_capacity_ah")

    return {
        "nominal_capacity_ah": _json_scalar(nominal_capacity),
        "nominal_capacity_source": "user_provided" if nominal_capacity is not None else None,
        "chemistry": context.get("chemistry"),
        "chemistry_source": "user_provided" if context.get("chemistry") else None,
        "capacity_unit": reader.get("capacity_unit"),
        "time_unit": reader.get("time_unit"),
        "soc_unit": reader.get("soc_unit"),
        "current_sign_convention": (
            "positive_current_is_charge"
            if reader.get("positive_current_is_charge", True)
            else "negative_current_is_charge"
        ),
        "current_rest_threshold_a": reader.get("current_rest_threshold_a"),
        "observed_voltage_window_v": _min_max(normalized, "voltage_v")
        if normalized is not None
        else None,
        "observed_discharge_capacity_range_ah": _min_max(
            cycle_features,
            "discharge_capacity_ah",
        )
        if cycle_features is not None
        else None,
        "observed_charge_capacity_range_ah": _min_max(
            cycle_features,
            "charge_capacity_ah",
        )
        if cycle_features is not None
        else None,
    }


def _build_dataset_overview(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    normalized = tables.get("normalized_timeseries")
    cycle_features = tables.get("cycle_features")
    overview: dict[str, Any] = {
        "available_feature_tables": {
            name: {"rows": int(len(frame)), "columns": int(len(frame.columns))}
            for name, frame in tables.items()
            if name != "normalized_timeseries"
        }
    }

    cycle_source = cycle_features if cycle_features is not None else normalized
    if cycle_source is not None and "cycle_index" in cycle_source.columns:
        cycles = cycle_source["cycle_index"].dropna()
        if not cycles.empty:
            overview["cycle_count"] = int(cycles.nunique())
            overview["cycle_min"] = int(cycles.min())
            overview["cycle_max"] = int(cycles.max())

    if normalized is not None and not normalized.empty:
        overview["raw_timeseries_rows"] = int(len(normalized))
        overview["step_type_counts"] = _value_counts(normalized, "step_type")
        overview["time_span_h"] = _range_span(normalized, "time_s", scale=3600.0)
        overview["voltage_range_v"] = _min_max(normalized, "voltage_v")
        overview["current_range_a"] = _min_max(normalized, "current_a")
        overview["temperature_range_c"] = _min_max(normalized, "temperature_c")
        overview["capacity_columns_present"] = {
            "charge_capacity_ah": _has_non_null(normalized, "charge_capacity_ah"),
            "discharge_capacity_ah": _has_non_null(normalized, "discharge_capacity_ah"),
        }
    return _clean_record(overview)


def _build_quality_summary(
    tables: dict[str, pd.DataFrame],
    cell_context: dict[str, Any],
    analysis_config: dict[str, Any],
) -> dict[str, Any]:
    cycle_features = tables.get("cycle_features")
    delta_q_features = tables.get("delta_q_features")
    normalized = tables.get("normalized_timeseries")
    warnings: list[dict[str, Any]] = []

    cycle_quality = _cycle_completeness_summary(cycle_features)
    capacity_quality = _capacity_reliability_summary(cycle_features, cell_context)
    reference_target = _reference_target_summary(cycle_features, delta_q_features, analysis_config)
    feature_computability = _feature_computability_summary(tables, cell_context)

    if cycle_quality.get("usable_cycle_count") == 0 and cycle_quality.get("cycle_count"):
        warnings.append(
            {
                "severity": "high",
                "issue": "no_usable_complete_cycles",
                "detail": (
                    "No cycles passed the basic charge/discharge completeness checks; capacity "
                    "and trend interpretation should not be trusted."
                ),
            }
        )
    elif cycle_quality.get("partial_or_unbalanced_cycle_count", 0) > 0:
        warnings.append(
            {
                "severity": "medium",
                "issue": "partial_or_unbalanced_cycles_detected",
                "detail": (
                    f"{cycle_quality['partial_or_unbalanced_cycle_count']} cycle(s) appear partial "
                    "or charge/discharge imbalanced and should be treated cautiously."
                ),
            }
        )

    if capacity_quality.get("status") == "low_confidence":
        warnings.append(
            {
                "severity": "medium",
                "issue": "capacity_reliability_low",
                "detail": capacity_quality.get(
                    "reason",
                    "Capacity reliability checks found issues.",
                ),
            }
        )

    if not reference_target.get("reference_cycle_present", True):
        warnings.append(
            {
                "severity": "medium",
                "issue": "reference_cycle_missing",
                "detail": (
                    f"Configured reference cycle {reference_target.get('reference_cycle')} "
                    "was not found in cycle features."
                ),
            }
        )
    if not reference_target.get("target_cycle_present", True):
        warnings.append(
            {
                "severity": "medium",
                "issue": "target_cycle_missing",
                "detail": (
                    f"Configured target cycle {reference_target.get('target_cycle')} "
                    "was not found in cycle features."
                ),
            }
        )
    if reference_target.get("reference_cycle_present") and not reference_target.get(
        "reference_cycle_usable",
        True,
    ):
        warnings.append(
            {
                "severity": "medium",
                "issue": "reference_cycle_not_usable",
                "detail": (
                    f"Configured reference cycle {reference_target.get('reference_cycle')} "
                    "is present but failed basic completeness or capacity-balance checks."
                ),
            }
        )
    if reference_target.get("target_cycle_present") and not reference_target.get(
        "target_cycle_usable",
        True,
    ):
        warnings.append(
            {
                "severity": "medium",
                "issue": "target_cycle_not_usable",
                "detail": (
                    f"Configured target cycle {reference_target.get('target_cycle')} "
                    "is present but failed basic completeness or capacity-balance checks."
                ),
            }
        )
    if reference_target.get("target_not_after_reference"):
        warnings.append(
            {
                "severity": "medium",
                "issue": "target_cycle_not_after_reference",
                "detail": "Configured target cycle is not later than the reference cycle.",
            }
        )
    if reference_target.get("delta_q_expected") and not reference_target.get("delta_q_available"):
        warnings.append(
            {
                "severity": "medium",
                "issue": "delta_q_not_computed_for_configured_cycles",
                "detail": (
                    "Delta-Q features were not computed for the configured reference/target cycles; "
                    "one or both cycles may lack enough discharge curve points or overlapping voltage range."
                ),
            }
        )

    if not feature_computability.get("c_rate_computable", False):
        warnings.append(
            {
                "severity": "info",
                "issue": "c_rate_features_limited",
                "detail": "C-rate features require nominal capacity; without it, C-rate fields are null.",
            }
        )
    if not feature_computability.get("temperature_available", False):
        warnings.append(
            {
                "severity": "info",
                "issue": "thermal_features_limited",
                "detail": "Temperature-derived features are unavailable because temperature data is missing.",
            }
        )

    return _clean_record(
        {
            "cycle_completeness": cycle_quality,
            "capacity_reliability": capacity_quality,
            "reference_target_cycles": reference_target,
            "feature_computability": feature_computability,
            "normalized_timeseries_available": normalized is not None and not normalized.empty,
            "warnings": warnings,
        }
    )


def _cycle_completeness_summary(cycle_features: pd.DataFrame | None) -> dict[str, Any]:
    if cycle_features is None or cycle_features.empty:
        return {
            "cycle_count": 0,
            "usable_cycle_count": 0,
            "complete_cycle_count": 0,
            "partial_or_unbalanced_cycle_count": 0,
            "partial_or_unbalanced_cycles": [],
        }

    frame = cycle_features.sort_values("cycle_index").copy()
    usable_mask = _usable_cycle_mask(frame)
    complete_mask = _complete_cycle_mask(frame)
    cycle_values = frame["cycle_index"] if "cycle_index" in frame.columns else pd.Series(range(len(frame)))
    bad_cycles = cycle_values[~usable_mask].dropna().astype(int).tolist()
    usable_cycles = cycle_values[usable_mask].dropna().astype(int).tolist()

    return {
        "cycle_count": int(len(frame)),
        "complete_cycle_count": int(complete_mask.sum()),
        "usable_cycle_count": int(usable_mask.sum()),
        "first_usable_cycle": usable_cycles[0] if usable_cycles else None,
        "last_usable_cycle": usable_cycles[-1] if usable_cycles else None,
        "partial_or_unbalanced_cycle_count": int((~usable_mask).sum()),
        "partial_or_unbalanced_cycles": bad_cycles[:20],
        "completion_rule": (
            "charge/discharge duration must be present, charge and discharge capacity must be "
            f"positive, and discharge/charge capacity ratio must be between "
            f"{_CAPACITY_BALANCE_MIN} and {_CAPACITY_BALANCE_MAX}."
        ),
    }


def _capacity_reliability_summary(
    cycle_features: pd.DataFrame | None,
    cell_context: dict[str, Any],
) -> dict[str, Any]:
    if cycle_features is None or cycle_features.empty:
        return {"status": "unavailable", "reason": "No cycle_features table is available."}

    frame = cycle_features.sort_values("cycle_index").copy()
    usable = frame[_usable_cycle_mask(frame)]
    if usable.empty:
        return {
            "status": "low_confidence",
            "reason": "No cycles passed charge/discharge balance checks.",
            "usable_cycle_count": 0,
        }

    ce_values = (
        pd.to_numeric(usable["coulombic_efficiency"], errors="coerce")
        if "coulombic_efficiency" in usable.columns
        else pd.Series(dtype=float)
    )
    abnormal_ce = ce_values[(ce_values < _CE_MIN) | (ce_values > _CE_MAX)].dropna()
    discharge = (
        pd.to_numeric(usable["discharge_capacity_ah"], errors="coerce").dropna()
        if "discharge_capacity_ah" in usable.columns
        else pd.Series(dtype=float)
    )
    nominal_capacity = cell_context.get("nominal_capacity_ah")

    status = "high_confidence"
    reasons: list[str] = []
    if len(usable) < 3:
        status = "low_confidence"
        reasons.append("Fewer than 3 usable cycles are available.")
    if len(abnormal_ce) > 0:
        status = "low_confidence"
        reasons.append(f"{len(abnormal_ce)} usable cycle(s) have coulombic efficiency outside 0.95-1.05.")
    if discharge.empty:
        status = "low_confidence"
        reasons.append("No usable discharge capacity values are available.")

    result = {
        "status": status,
        "reason": " ".join(reasons) if reasons else "Usable cycles pass basic capacity checks.",
        "usable_cycle_count": int(len(usable)),
        "abnormal_coulombic_efficiency_count": int(len(abnormal_ce)),
        "median_coulombic_efficiency": _json_scalar(ce_values.dropna().median())
        if not ce_values.dropna().empty
        else None,
        "discharge_capacity_range_ah": _min_max(usable, "discharge_capacity_ah"),
    }
    if nominal_capacity and not discharge.empty:
        result["discharge_capacity_vs_nominal_range_percent"] = {
            "min": _json_scalar(100.0 * discharge.min() / float(nominal_capacity)),
            "max": _json_scalar(100.0 * discharge.max() / float(nominal_capacity)),
        }
    else:
        result["discharge_capacity_vs_nominal_range_percent"] = None
    return result


def _reference_target_summary(
    cycle_features: pd.DataFrame | None,
    delta_q_features: pd.DataFrame | None,
    analysis_config: dict[str, Any],
) -> dict[str, Any]:
    key_params = analysis_config.get("key_parameters", {})
    reference_cycle = key_params.get("reference_cycle")
    target_cycle = key_params.get("target_cycle")
    result = {
        "reference_cycle": reference_cycle,
        "target_cycle": target_cycle,
        "reference_cycle_present": None,
        "target_cycle_present": None,
        "reference_cycle_usable": None,
        "target_cycle_usable": None,
        "target_not_after_reference": (
            reference_cycle is not None and target_cycle is not None and target_cycle <= reference_cycle
        ),
        "delta_q_expected": reference_cycle is not None and target_cycle is not None,
        "delta_q_available": delta_q_features is not None and not delta_q_features.empty,
    }

    if cycle_features is None or cycle_features.empty or "cycle_index" not in cycle_features.columns:
        return result

    frame = cycle_features.sort_values("cycle_index").copy()
    cycles = set(frame["cycle_index"].dropna().astype(int).tolist())
    usable_cycles = set(frame.loc[_usable_cycle_mask(frame), "cycle_index"].dropna().astype(int).tolist())
    if reference_cycle is not None:
        result["reference_cycle_present"] = int(reference_cycle) in cycles
        result["reference_cycle_usable"] = int(reference_cycle) in usable_cycles
    if target_cycle is not None:
        result["target_cycle_present"] = int(target_cycle) in cycles
        result["target_cycle_usable"] = int(target_cycle) in usable_cycles
    return result


def _feature_computability_summary(
    tables: dict[str, pd.DataFrame],
    cell_context: dict[str, Any],
) -> dict[str, Any]:
    normalized = tables.get("normalized_timeseries")
    stress = tables.get("stress_features")
    return {
        "c_rate_computable": cell_context.get("nominal_capacity_ah") is not None,
        "temperature_available": _has_non_null(normalized, "temperature_c")
        if normalized is not None
        else False,
        "delta_q_available": tables.get("delta_q_features") is not None
        and not tables["delta_q_features"].empty,
        "ica_dva_available": tables.get("ica_dva_features") is not None
        and not tables["ica_dva_features"].empty,
        "relaxation_available": tables.get("relaxation_features") is not None
        and not tables["relaxation_features"].empty,
        "stress_available": stress is not None and not stress.empty,
    }


def _build_data_quality(
    tables: dict[str, pd.DataFrame],
    tags: list[dict[str, Any]],
    cell_context: dict[str, Any],
    analysis_config: dict[str, Any],
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    normalized = tables.get("normalized_timeseries")
    cycle_features = tables.get("cycle_features")
    quality_summary = _build_quality_summary(tables, cell_context, analysis_config)

    if cell_context.get("nominal_capacity_ah") is None:
        warnings.append(
            {
                "severity": "info",
                "issue": "nominal_capacity_missing",
                "detail": (
                    "Nominal capacity was not provided, so SOH relative to nameplate capacity "
                    "was not computed."
                ),
            }
        )

    if cell_context.get("chemistry") is None:
        warnings.append(
            {
                "severity": "info",
                "issue": "chemistry_missing",
                "detail": (
                    "Cell chemistry was not provided; chemistry-specific interpretation should "
                    "be inferred only from external context, not from the file name."
                ),
            }
        )

    cycle_count = None
    if cycle_features is not None and "cycle_index" in cycle_features.columns:
        cycle_count = int(cycle_features["cycle_index"].nunique())
    elif normalized is not None and "cycle_index" in normalized.columns:
        cycle_count = int(normalized["cycle_index"].nunique())

    if cycle_count is not None:
        if cycle_count < 10:
            warnings.append(
                {
                    "severity": "high",
                    "issue": "very_short_cycle_series",
                    "detail": (
                        f"Only {cycle_count} cycles are available; degradation trends are weak "
                        "and should be treated as preliminary."
                    ),
                }
            )
        elif cycle_count < 30:
            warnings.append(
                {
                    "severity": "medium",
                    "issue": "short_cycle_series",
                    "detail": (
                        f"{cycle_count} cycles are available; long-term aging conclusions are limited."
                    ),
                }
            )

    if normalized is not None:
        if not _has_non_null(normalized, "temperature_c"):
            warnings.append(
                {
                    "severity": "medium",
                    "issue": "temperature_missing",
                    "detail": "Temperature was not present or was empty, so thermal stress cannot be assessed.",
                }
            )
        missing_required = [
            column
            for column in ("time_s", "voltage_v", "current_a")
            if column in normalized.columns and normalized[column].isna().any()
        ]
        if missing_required:
            warnings.append(
                {
                    "severity": "medium",
                    "issue": "missing_required_values",
                    "detail": f"Missing values were detected in: {', '.join(missing_required)}.",
                }
            )

    if cycle_features is not None and not cycle_features.empty:
        first_cycle_warning = _first_cycle_balance_warning(cycle_features)
        if first_cycle_warning:
            warnings.append(first_cycle_warning)
        ce_warning = _efficiency_warning(cycle_features)
        if ce_warning:
            warnings.append(ce_warning)

    for warning in quality_summary.get("warnings", []):
        warnings.append(warning)

    if not tags:
        warnings.append(
            {
                "severity": "info",
                "issue": "no_degradation_tags",
                "detail": "No rule-based degradation evidence tags were generated for this cell.",
            }
        )

    status = "ok"
    severities = {str(warning["severity"]) for warning in warnings}
    if "high" in severities:
        status = "use_with_caution"
    elif "medium" in severities:
        status = "usable_with_limitations"

    return {
        "status": status,
        "quality_summary": quality_summary,
        "warnings": warnings,
        "assumptions": [
            "Cycle-level conclusions are computed from the normalized BDS/cycler export.",
            "Diagnostic tags are evidence signals, not definitive root-cause labels.",
        ],
    }


def _build_cycle_life_summary(
    cycle_features: pd.DataFrame | None,
    cell_context: dict[str, Any],
) -> dict[str, Any]:
    if cycle_features is None or cycle_features.empty:
        return {}

    frame = cycle_features.sort_values("cycle_index").copy()
    valid = frame[_usable_cycle_mask(frame)]
    if valid.empty and "discharge_capacity_ah" in frame.columns:
        valid = frame[frame["discharge_capacity_ah"].notna() & (frame["discharge_capacity_ah"] > 0)]
    if valid.empty:
        valid = frame

    start = _first_complete_cycle(valid)
    end = valid.iloc[-1]
    summary: dict[str, Any] = {
        "cycle_count": int(frame["cycle_index"].nunique()) if "cycle_index" in frame.columns else int(len(frame)),
        "first_cycle": _cycle_record(start),
        "last_cycle": _cycle_record(end),
        "sampled_cycle_records": _sample_cycle_records(frame),
        "trends": {},
    }

    if "cycle_index" in valid.columns:
        for column in (
            "discharge_capacity_ah",
            "charge_capacity_ah",
            "coulombic_efficiency",
            "energy_efficiency",
            "cv_capacity_fraction",
        ):
            trend = _linear_trend(valid, "cycle_index", column)
            if trend:
                summary["trends"][column] = trend

    start_cap = _get_number(start, "discharge_capacity_ah")
    end_cap = _get_number(end, "discharge_capacity_ah")
    if start_cap and end_cap is not None:
        summary["capacity_change"] = {
            "start_cycle": _json_scalar(start.get("cycle_index")),
            "end_cycle": _json_scalar(end.get("cycle_index")),
            "start_discharge_capacity_ah": start_cap,
            "end_discharge_capacity_ah": end_cap,
            "absolute_change_ah": _json_scalar(end_cap - start_cap),
            "retention_fraction": _json_scalar(end_cap / start_cap),
            "retention_percent": _json_scalar(100.0 * end_cap / start_cap),
        }
        nominal_capacity = cell_context.get("nominal_capacity_ah")
        if nominal_capacity:
            nominal = float(nominal_capacity)
            summary["soh_vs_nominal"] = {
                "nominal_capacity_ah": nominal,
                "start_discharge_capacity_ah": start_cap,
                "last_discharge_capacity_ah": end_cap,
                "start_capacity_vs_nominal_fraction": _json_scalar(start_cap / nominal),
                "start_capacity_vs_nominal_percent": _json_scalar(100.0 * start_cap / nominal),
                "last_capacity_vs_nominal_fraction": _json_scalar(end_cap / nominal),
                "last_capacity_vs_nominal_percent": _json_scalar(100.0 * end_cap / nominal),
                "absolute_change_vs_nominal_fraction": _json_scalar((end_cap - start_cap) / nominal),
                "absolute_change_vs_nominal_percent": _json_scalar(
                    100.0 * (end_cap - start_cap) / nominal
                ),
            }
        else:
            summary["soh_vs_nominal"] = None
    return _clean_record(summary)


def _build_feature_highlights(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return _clean_record(
        {
            "stress_summary": _stress_summary(tables.get("stress_features")),
            "delta_q_summary": _delta_q_summary(tables.get("delta_q_features")),
            "ica_dva_summary": _ica_dva_summary(tables.get("ica_dva_features")),
            "relaxation_summary": _relaxation_summary(tables.get("relaxation_features")),
        }
    )


def _stress_summary(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    row = frame.iloc[0]
    keys = (
        "calendar_time_h",
        "throughput_ah",
        "equivalent_full_cycles",
        "high_soc_rest_fraction",
        "rest_fraction",
        "charge_fraction",
        "discharge_fraction",
        "max_instant_discharge_c_rate",
        "current_variance_a2",
        "c_rate_variance",
        "soc_mean",
        "soc_min",
        "soc_max",
        "voltage_v_min",
        "voltage_v_max",
        "temperature_c_mean",
    )
    summary = {key: _json_scalar(row.get(key)) for key in keys if key in row.index}
    summary["dominant_soc_bins"] = _histogram_top_bins(row, "soc_hist_bin", top_n=3)
    summary["dominant_voltage_bins"] = _histogram_top_bins(row, "voltage_v_hist_bin", top_n=3)
    summary["dominant_c_rate_bins"] = _histogram_top_bins(row, "c_rate_hist_bin", top_n=3)
    return summary


def _delta_q_summary(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    row = frame.iloc[0]
    keys = (
        "reference_cycle",
        "target_cycle",
        "voltage_min_v",
        "voltage_max_v",
        "delta_q_area_ah_v",
        "delta_q_abs_area_ah_v",
        "delta_q_l1",
        "delta_q_l2",
        "delta_q_variance",
        "delta_q_min_voltage_v",
        "delta_q_max_voltage_v",
        "delta_q_ah_mean",
        "delta_q_ah_min",
        "delta_q_ah_max",
    )
    windows = {
        f"window_{idx}": {
            "mean": _json_scalar(row.get(f"delta_q_window_{idx}_mean")),
            "variance": _json_scalar(row.get(f"delta_q_window_{idx}_variance")),
            "min": _json_scalar(row.get(f"delta_q_window_{idx}_min")),
            "max": _json_scalar(row.get(f"delta_q_window_{idx}_max")),
        }
        for idx in range(1, 6)
        if f"delta_q_window_{idx}_mean" in row.index
    }
    return {
        "comparison": {key: _json_scalar(row.get(key)) for key in keys if key in row.index},
        "voltage_window_summaries": windows,
    }


def _ica_dva_summary(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    frame = frame.sort_values("cycle_index") if "cycle_index" in frame.columns else frame
    first = frame.iloc[0]
    last = frame.iloc[-1]
    return {
        "row_count": int(len(frame)),
        "first_cycle": _ica_dva_cycle_record(first),
        "last_cycle": _ica_dva_cycle_record(last),
        "trends": {
            key: trend
            for key in (
                "ica_area",
                "dva_area",
                "ica_peak_count",
                "dva_peak_count",
                "ica_peak_1_x",
                "ica_peak_1_height",
            )
            if (trend := _linear_trend(frame, "cycle_index", key))
        },
        "sampled_cycles": [_ica_dva_cycle_record(row) for _, row in _sample_rows(frame).iterrows()],
    }


def _relaxation_summary(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    frame = frame.sort_values("cycle_index") if "cycle_index" in frame.columns else frame
    keys = (
        "cycle_index",
        "step_index",
        "rest_duration_s",
        "rest_voltage_initial_v",
        "rest_voltage_final_v",
        "rest_voltage_delta_v",
        "rest_voltage_at_30s_v",
        "rest_voltage_at_300s_v",
        "rest_exp_tau_s",
        "rest_exp_amplitude_v",
        "rest_exp_rmse_v",
    )
    return {
        "row_count": int(len(frame)),
        "first_rest": _select_keys(frame.iloc[0], keys),
        "last_rest": _select_keys(frame.iloc[-1], keys),
        "trends": {
            key: trend
            for key in (
                "rest_exp_tau_s",
                "rest_voltage_delta_v",
                "rest_voltage_initial_slope_v_per_s",
                "rest_voltage_linear_slope_v_per_s",
            )
            if (trend := _linear_trend(frame, "cycle_index", key))
        },
        "sampled_rest_records": [_select_keys(row, keys) for _, row in _sample_rows(frame).iterrows()],
    }


def _build_summary(record: dict[str, Any]) -> dict[str, Any]:
    overview = record.get("dataset_overview", {})
    cycle = record.get("cycle_life_summary", {})
    cell_context = record.get("cell_context", {})
    data_quality = record.get("data_quality", {})
    diagnostics = record.get("diagnostic_evidence", [])
    stress = record.get("feature_highlights", {}).get("stress_summary", {})

    summary = {
        "cell_id": record.get("cell_id"),
        "nominal_capacity_ah": cell_context.get("nominal_capacity_ah"),
        "chemistry": cell_context.get("chemistry"),
        "cycle_count": overview.get("cycle_count"),
        "observed_time_h": overview.get("time_span_h"),
        "data_quality_status": data_quality.get("status"),
        "main_capacity_result": cycle.get("capacity_change", {}),
        "soh_vs_nominal": cycle.get("soh_vs_nominal"),
        "diagnostic_signal_count": len(diagnostics),
        "diagnostic_signals": [tag.get("signal") for tag in diagnostics],
        "stress_snapshot": {
            key: stress.get(key)
            for key in (
                "calendar_time_h",
                "throughput_ah",
                "equivalent_full_cycles",
                "high_soc_rest_fraction",
                "max_instant_discharge_c_rate",
            )
            if key in stress
        },
    }
    warnings = data_quality.get("warnings", [])
    summary["most_important_warnings"] = warnings[:3]
    return _clean_record(summary)


def _build_review_notes(record: dict[str, Any]) -> dict[str, Any]:
    warnings = record.get("data_quality", {}).get("warnings", [])
    high_warnings = [warning for warning in warnings if warning.get("severity") == "high"]
    medium_warnings = [warning for warning in warnings if warning.get("severity") == "medium"]
    diagnostics = record.get("diagnostic_evidence", [])
    notes = [
        "Use cycle_life_summary for capacity retention and efficiency trends.",
        "Use feature_highlights for Delta-Q, ICA/DVA, relaxation, and stress evidence.",
        "Use diagnostic_evidence as rule-based evidence signals; combine with domain knowledge before assigning root cause.",
    ]
    if high_warnings or medium_warnings:
        notes.append("Respect data_quality warnings before making strong degradation claims.")
    if not diagnostics:
        notes.append("Absence of degradation tags does not prove absence of degradation; it means no implemented rule fired.")
    return {
        "recommended_use": notes,
        "review_targets": [
            "capacity fade or retention",
            "coulombic and energy efficiency behavior",
            "voltage-window-specific Delta-Q changes",
            "ICA/DVA peak shifts or peak-count changes",
            "relaxation time constant and rest-voltage recovery",
            "usage stress: SOC exposure, voltage window, C-rate, throughput, temperature if available",
        ],
        "limits": [
            warning["detail"] for warning in warnings if warning.get("severity") in {"high", "medium"}
        ],
    }


def _first_complete_cycle(frame: pd.DataFrame) -> pd.Series:
    required = {"charge_capacity_ah", "discharge_capacity_ah"}
    if not required.issubset(frame.columns):
        return frame.iloc[0]
    usable = frame[_usable_cycle_mask(frame)]
    if not usable.empty:
        return usable.iloc[0]
    for _, row in frame.iterrows():
        charge = _get_number(row, "charge_capacity_ah")
        discharge = _get_number(row, "discharge_capacity_ah")
        if charge and discharge and _CAPACITY_BALANCE_MIN <= discharge / charge <= _CAPACITY_BALANCE_MAX:
            return row
    return frame.iloc[0]


def _complete_cycle_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in ("charge_duration_s", "discharge_duration_s", "charge_capacity_ah", "discharge_capacity_ah"):
        if column not in frame.columns:
            return pd.Series(False, index=frame.index)
    mask &= pd.to_numeric(frame["charge_duration_s"], errors="coerce").fillna(0) > 0
    mask &= pd.to_numeric(frame["discharge_duration_s"], errors="coerce").fillna(0) > 0
    mask &= pd.to_numeric(frame["charge_capacity_ah"], errors="coerce").fillna(0) > 0
    mask &= pd.to_numeric(frame["discharge_capacity_ah"], errors="coerce").fillna(0) > 0
    return mask


def _usable_cycle_mask(frame: pd.DataFrame) -> pd.Series:
    complete = _complete_cycle_mask(frame)
    if "charge_capacity_ah" not in frame.columns or "discharge_capacity_ah" not in frame.columns:
        return complete
    charge = pd.to_numeric(frame["charge_capacity_ah"], errors="coerce")
    discharge = pd.to_numeric(frame["discharge_capacity_ah"], errors="coerce")
    ratio = discharge / charge.replace(0, np.nan)
    balanced = ratio.between(_CAPACITY_BALANCE_MIN, _CAPACITY_BALANCE_MAX).fillna(False)
    return complete & balanced


def _first_cycle_balance_warning(frame: pd.DataFrame) -> dict[str, Any] | None:
    required = {"cycle_index", "charge_capacity_ah", "discharge_capacity_ah"}
    if not required.issubset(frame.columns) or frame.empty:
        return None
    row = frame.sort_values("cycle_index").iloc[0]
    charge = _get_number(row, "charge_capacity_ah")
    discharge = _get_number(row, "discharge_capacity_ah")
    if not charge or not discharge:
        return None
    ratio = discharge / charge
    if _CAPACITY_BALANCE_MIN <= ratio <= _CAPACITY_BALANCE_MAX:
        return None
    return {
        "severity": "medium",
        "issue": "first_cycle_charge_discharge_imbalance",
        "detail": (
            f"Cycle {int(row['cycle_index'])} charge/discharge capacity ratio is unusual "
            f"(charge={charge:.4g} Ah, discharge={discharge:.4g} Ah, discharge/charge={ratio:.3g}). "
            "The first cycle may be partial or include formation/rest carryover."
        ),
    }


def _efficiency_warning(frame: pd.DataFrame) -> dict[str, Any] | None:
    if "coulombic_efficiency" not in frame.columns:
        return None
    # Evaluate only usable (complete and capacity-balanced) cycles so this warning agrees with
    # data_quality.quality_summary.capacity_reliability; partial/formation cycles are already
    # surfaced via the cycle-completeness checks and should not double-count here.
    usable = frame[_usable_cycle_mask(frame)]
    values = usable["coulombic_efficiency"].dropna()
    if values.empty:
        return None
    outside = values[(values < _CE_MIN) | (values > _CE_MAX)]
    if outside.empty:
        return None
    return {
        "severity": "medium",
        "issue": "coulombic_efficiency_outside_expected_range",
        "detail": (
            f"{len(outside)} usable cycle(s) have coulombic efficiency outside "
            f"{_CE_MIN}-{_CE_MAX}; review whether these cycles are reliable before interpretation."
        ),
    }


def _cycle_record(row: pd.Series) -> dict[str, Any]:
    return _select_keys(
        row,
        (
            "cycle_index",
            "n_points",
            "duration_s",
            "charge_duration_s",
            "discharge_duration_s",
            "rest_duration_s",
            "charge_capacity_ah",
            "discharge_capacity_ah",
            "charge_energy_wh",
            "discharge_energy_wh",
            "coulombic_efficiency",
            "energy_efficiency",
            "specific_throughput_efc",
            "cv_capacity_fraction",
            "voltage_v_min",
            "voltage_v_max",
            "current_a_min",
            "current_a_max",
            "temperature_c_mean",
        ),
    )


def _sample_cycle_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_cycle_record(row) for _, row in _sample_rows(frame).iterrows()]


def _ica_dva_cycle_record(row: pd.Series) -> dict[str, Any]:
    return _select_keys(
        row,
        (
            "cycle_index",
            "ica_area",
            "dva_area",
            "ica_peak_count",
            "dva_peak_count",
            "ica_peak_1_x",
            "ica_peak_1_height",
            "ica_peak_1_prominence",
            "ica_peak_1_width_x",
            "ica_peak_2_x",
            "ica_peak_2_height",
            "dva_peak_1_x",
            "dva_peak_1_height",
            "dva_peak_1_prominence",
            "dva_peak_2_x",
            "dva_peak_2_height",
        ),
    )


def _sample_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) <= _SAMPLE_CYCLE_LIMIT:
        return frame
    head = frame.head(_SAMPLE_CYCLE_LIMIT // 2)
    tail = frame.tail(_SAMPLE_CYCLE_LIMIT // 2)
    return pd.concat([head, tail]).drop_duplicates()


def _select_keys(row: pd.Series, keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: _json_scalar(row.get(key)) for key in keys if key in row.index}


def _linear_trend(frame: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any] | None:
    if x_col not in frame.columns or y_col not in frame.columns:
        return None
    subset = frame[[x_col, y_col]].dropna()
    if len(subset) < _TREND_MIN_POINTS:
        return None
    x = subset[x_col].astype(float).to_numpy()
    y = subset[y_col].astype(float).to_numpy()
    if np.unique(x).size < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    start = float(y[0])
    end = float(y[-1])
    return {
        "n_points": int(len(subset)),
        "start": _json_scalar(start),
        "end": _json_scalar(end),
        "absolute_change": _json_scalar(end - start),
        "relative_change_fraction": _json_scalar((end - start) / start) if start else None,
        "slope_per_cycle": _json_scalar(slope),
        "intercept": _json_scalar(intercept),
    }


def _histogram_top_bins(row: pd.Series, prefix: str, *, top_n: int) -> list[dict[str, Any]]:
    bins = []
    idx = 0
    while f"{prefix}_{idx}_fraction" in row.index:
        fraction = row.get(f"{prefix}_{idx}_fraction")
        if pd.notna(fraction):
            bins.append(
                {
                    "bin_index": idx,
                    "fraction": _json_scalar(fraction),
                    "low": _json_scalar(row.get(f"{prefix}_{idx}_low")),
                    "high": _json_scalar(row.get(f"{prefix}_{idx}_high")),
                }
            )
        idx += 1
    return sorted(bins, key=lambda item: item["fraction"] or 0.0, reverse=True)[:top_n]


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).items()}


def _min_max(frame: pd.DataFrame, column: str) -> dict[str, Any] | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return {"min": _json_scalar(values.min()), "max": _json_scalar(values.max())}


def _range_span(frame: pd.DataFrame, column: str, *, scale: float) -> float | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return _json_scalar((values.max() - values.min()) / scale)


def _has_non_null(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and frame[column].notna().any()


def _get_number(row: pd.Series, key: str) -> float | None:
    if key not in row.index:
        return None
    value = row.get(key)
    if pd.isna(value):
        return None
    return float(value)


def _signal_interpretation(signal: str) -> str:
    hints = {
        "capacity_fade": "Capacity decreases across cycles; consider loss of lithium inventory, active material loss, impedance growth, or protocol artifacts.",
        "ica_peak_shift": "ICA peak movement can indicate electrode stoichiometry shifts, loss of lithium inventory, or changing reaction heterogeneity.",
        "relaxation_tau_increase": "Increasing relaxation time can be consistent with resistance growth, slower transport, or stronger concentration polarization.",
        "high_soc_rest_exposure": "High-SOC rest exposure is a calendar-aging stressor and can accelerate side reactions for many Li-ion chemistries.",
        "dynamic_current_variance": "High current variability may indicate dynamic load stress and should be interpreted with C-rate and thermal context.",
        "high_instantaneous_discharge_rate": "Discharge C-rate approaches or exceeds a provided datasheet limit, raising stress and heating concerns.",
    }
    return hints.get(signal, "Use this signal as supporting evidence and combine it with cycle, stress, ICA/DVA, and relaxation summaries.")


def _clean_record(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_record(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_clean_record(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_clean_record(v) for v in value]
    return _json_scalar(value)


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
