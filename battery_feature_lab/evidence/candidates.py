"""Convert feature tables and diagnostic tags into evidence candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from battery_feature_lab.evidence.schema import empty_evidence_frame, normalize_evidence_frame


def build_evidence_candidates(
    tables: dict[str, pd.DataFrame],
    degradation_tags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build evidence candidates from BFL feature tables.

    The first-phase evidence layer is intentionally conservative: each candidate is a measured
    fact, derived feature, or rule-based diagnostic signal with source metadata and an
    interpretation boundary. Protocol metadata is joined from ``protocol_segments`` after
    candidate construction so every evidence path receives the same cycle context.
    """

    records: list[dict[str, Any]] = []
    records.extend(_cycle_feature_candidates(tables.get("cycle_features")))
    records.extend(_delta_q_candidates(tables.get("delta_q_features")))
    records.extend(_ica_dva_candidates(tables.get("ica_dva_features")))
    records.extend(_relaxation_candidates(tables.get("relaxation_features")))
    records.extend(_stress_candidates(tables.get("stress_features")))
    records.extend(_diagnostic_tag_candidates(degradation_tags))
    records.extend(_generic_numeric_candidates(tables, records))
    records = _attach_protocol_metadata(records, tables.get("protocol_segments"))
    if not records:
        return empty_evidence_frame()
    return normalize_evidence_frame(pd.DataFrame.from_records(records))


def _attach_protocol_metadata(
    records: list[dict[str, Any]],
    protocol_segments: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Join cycle protocol context onto every evidence construction path."""

    if not records or protocol_segments is None or protocol_segments.empty:
        return records
    required = {"cell_id", "cycle_index", "protocol"}
    if not required.issubset(protocol_segments.columns):
        return records

    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    grouped = protocol_segments.groupby(["cell_id", "cycle_index"], sort=False)
    for (cell_id, cycle_index), group in grouped:
        ordered = (
            group.sort_values("segment_index")
            if "segment_index" in group.columns
            else group
        )
        first = ordered.iloc[0]
        lookup[(str(cell_id), int(cycle_index))] = {
            "protocol": _text_or_unknown(first.get("protocol")),
            "protocol_family": _text_or_unknown(first.get("protocol_family")),
            "protocol_confidence": _number(first.get("protocol_confidence")),
        }

    for record in records:
        cell_id = str(record.get("cell_id"))
        cycle_index = _int_or_none(record.get("cycle_index"))
        cycle_start = _int_or_none(record.get("cycle_start"))
        cycle_end = _int_or_none(record.get("cycle_end"))
        if cycle_index is not None:
            contexts = [lookup.get((cell_id, cycle_index))]
        elif cycle_start is not None and cycle_end is not None:
            contexts = [
                context
                for (candidate_cell, candidate_cycle), context in lookup.items()
                if candidate_cell == cell_id and cycle_start <= candidate_cycle <= cycle_end
            ]
        else:
            contexts = []
        contexts = [context for context in contexts if context is not None]
        if not contexts:
            continue

        protocols = {context["protocol"] for context in contexts}
        families = {context["protocol_family"] for context in contexts}
        confidences = [
            context["protocol_confidence"]
            for context in contexts
            if context["protocol_confidence"] is not None
        ]
        record["protocol"] = protocols.pop() if len(protocols) == 1 else "mixed"
        record["protocol_family"] = families.pop() if len(families) == 1 else "mixed"
        record["protocol_confidence"] = min(confidences) if confidences else None
    return records


def _text_or_unknown(value: Any) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return "unknown"
    return str(value)


def _generic_numeric_candidates(
    tables: dict[str, pd.DataFrame],
    existing_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose numeric outputs from current and future feature tables without a column whitelist."""

    ignored_tables = {
        "normalized_timeseries",
        "degradation_tags",
        "evidence_candidates",
        "selected_evidence",
        "protocol_segments",
    }
    structural_columns = {
        "cell_id",
        "cycle_index",
        "reference_cycle",
        "target_cycle",
        "source_row_index",
        "selection_rank",
        "selected",
    }
    covered = {
        (str(record.get("source_table")), str(record.get("feature_name")))
        for record in existing_records
    }
    records: list[dict[str, Any]] = []

    for source_table, frame in sorted(tables.items()):
        if source_table in ignored_tables or frame is None or frame.empty:
            continue
        if "cell_id" not in frame.columns:
            continue

        numeric_columns = [
            column
            for column in frame.columns
            if column not in structural_columns
            and pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_bool_dtype(frame[column])
        ]
        for cell_id, group in frame.groupby("cell_id", sort=True):
            ordered = _sort_by_cycle(group)
            for column in numeric_columns:
                valid = ordered[pd.to_numeric(ordered[column], errors="coerce").notna()]
                if valid.empty:
                    continue
                output_name = f"{column}_change" if len(valid) > 1 else column
                if (source_table, output_name) in covered:
                    continue
                first = valid.iloc[0]
                last = valid.iloc[-1]
                first_value = _number(first.get(column))
                last_value = _number(last.get(column))
                if first_value is None or last_value is None:
                    continue

                if len(valid) > 1:
                    value = last_value - first_value
                    cycle_start = _int_or_none(first.get("cycle_index"))
                    cycle_end = _int_or_none(last.get("cycle_index"))
                    text = (
                        f"{column} changed from {first_value:.6g} to {last_value:.6g}"
                        f" between cycles {cycle_start} and {cycle_end}."
                    )
                    cycle_index = None
                    source_row_index = None
                else:
                    value = last_value
                    cycle_start = None
                    cycle_end = None
                    cycle_index = _int_or_none(last.get("cycle_index"))
                    source_row_index = _row_index(last)
                    text = f"{column} is {last_value:.6g} at cycle {cycle_index}."

                records.append(
                    _record(
                        cell_id=str(cell_id),
                        source_table=source_table,
                        source_row_index=source_row_index,
                        cycle_index=cycle_index,
                        cycle_start=cycle_start,
                        cycle_end=cycle_end,
                        feature_name=output_name,
                        evidence_type="numeric_feature",
                        value=value,
                        support_role="observed_numeric_feature",
                        reliability="unknown",
                        interpretation_hint=(
                            "This numeric feature is exposed without a mechanism interpretation; "
                            "its meaning follows the producing feature table."
                        ),
                        text=text,
                        protocol=_protocol_value(last),
                    )
                )
                covered.add((source_table, output_name))

    return records


def _cycle_feature_candidates(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records: list[dict[str, Any]] = []
    for cell_id, group in frame.groupby("cell_id", sort=True):
        ordered = _sort_by_cycle(group)
        first = _first_valid_row(ordered, "discharge_capacity_ah")
        last = _last_valid_row(ordered, "discharge_capacity_ah")
        if first is not None and last is not None:
            first_cap = _number(first.get("discharge_capacity_ah"))
            last_cap = _number(last.get("discharge_capacity_ah"))
            if first_cap and last_cap is not None:
                retention = last_cap / first_cap if first_cap else None
                records.append(
                    _record(
                        cell_id=str(cell_id),
                        source_table="cycle_features",
                        feature_name="discharge_capacity_retention",
                        evidence_type="derived_feature",
                        value=retention,
                        unit="fraction",
                        cycle_start=_int_or_none(first.get("cycle_index")),
                        cycle_end=_int_or_none(last.get("cycle_index")),
                        support_role="capacity_trend",
                        reliability="high",
                        interpretation_hint=(
                            "Capacity retention summarizes observed discharge-capacity change; "
                            "it supports fade/retention statements but not a unique mechanism."
                        ),
                        text=(
                            f"Discharge capacity changed from {first_cap:.4g} Ah at cycle "
                            f"{_int_or_none(first.get('cycle_index'))} to {last_cap:.4g} Ah "
                            "at cycle "
                            f"{_int_or_none(last.get('cycle_index'))}, retention={retention:.4g}."
                        ),
                    )
                )
        for column, role, unit, hint in (
            (
                "coulombic_efficiency",
                "efficiency",
                "fraction",
                "Coulombic efficiency helps assess cycle consistency and possible side reactions.",
            ),
            (
                "cv_capacity_fraction",
                "charge_protocol",
                "fraction",
                "CV capacity fraction can indicate changes in the charge tail or polarization.",
            ),
        ):
            first_metric = _first_valid_row(ordered, column)
            last_metric = _last_valid_row(ordered, column)
            for label, row in (("first", first_metric), ("last", last_metric)):
                if row is None:
                    continue
                value = _number(row.get(column))
                if value is None:
                    continue
                cycle = _int_or_none(row.get("cycle_index"))
                records.append(
                    _record(
                        cell_id=str(cell_id),
                        source_table="cycle_features",
                        source_row_index=_row_index(row),
                        cycle_index=cycle,
                        feature_name=column,
                        evidence_type="derived_feature",
                        value=value,
                        unit=unit,
                        support_role=role,
                        reliability="medium",
                        interpretation_hint=hint,
                        text=(
                            f"{label.capitalize()} observed {column} is {value:.4g} "
                            f"at cycle {cycle}."
                        ),
                    )
                )
    return records


def _delta_q_candidates(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        cell_id = str(row.get("cell_id"))
        ref = _int_or_none(row.get("reference_cycle"))
        target = _int_or_none(row.get("target_cycle"))
        for column, unit in (
            ("delta_q_abs_area_ah_v", "Ah*V"),
            ("delta_q_variance", "Ah^2"),
            ("delta_q_l2", "Ah"),
        ):
            value = _number(row.get(column))
            if value is None:
                continue
            records.append(
                _record(
                    cell_id=cell_id,
                    source_table="delta_q_features",
                    source_row_index=_row_index(row),
                    feature_name=column,
                    evidence_type="derived_feature",
                    value=value,
                    unit=unit,
                    cycle_start=ref,
                    cycle_end=target,
                    support_role="delta_q_change",
                    reliability="medium",
                    interpretation_hint=(
                        "Delta-Q differences localize charge/discharge curve changes between "
                        "reference and target cycles; they support change detection but not a "
                        "standalone mechanism diagnosis."
                    ),
                    text=(
                        f"{column} between reference cycle {ref} and target cycle {target} "
                        f"is {value:.4g}."
                    ),
                )
            )
    return records


def _ica_dva_candidates(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records: list[dict[str, Any]] = []
    for cell_id, group in frame.groupby("cell_id", sort=True):
        ordered = _sort_by_cycle(group)
        for column, unit, role in (
            ("ica_peak_1_x", "V", "ica_peak_shift"),
            ("ica_peak_1_height", "Ah/V", "ica_peak_shape"),
            ("ica_area", "Ah", "ica_area"),
            ("dva_area", "V/Ah", "dva_area"),
        ):
            first = _first_valid_row(ordered, column)
            last = _last_valid_row(ordered, column)
            if first is None or last is None:
                continue
            first_value = _number(first.get(column))
            last_value = _number(last.get(column))
            if first_value is None or last_value is None:
                continue
            first_cycle = _int_or_none(first.get("cycle_index"))
            last_cycle = _int_or_none(last.get("cycle_index"))
            records.append(
                _record(
                    cell_id=str(cell_id),
                    source_table="ica_dva_features",
                    feature_name=f"{column}_change",
                    evidence_type="derived_feature",
                    value=last_value - first_value,
                    unit=unit,
                    cycle_start=first_cycle,
                    cycle_end=last_cycle,
                    support_role=role,
                    reliability="medium",
                    interpretation_hint=(
                        "ICA/DVA peak and area changes can support electrode-balance or reaction "
                        "heterogeneity hypotheses, but should be combined with capacity and "
                        "protocol context."
                    ),
                    text=(
                        f"{column} changed from {first_value:.4g} at cycle {first_cycle} to "
                        f"{last_value:.4g} at cycle {last_cycle}."
                    ),
                )
            )
    return records


def _relaxation_candidates(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records: list[dict[str, Any]] = []
    for cell_id, group in frame.groupby("cell_id", sort=True):
        ordered = _sort_by_cycle(group)
        for column, unit, role in (
            ("rest_exp_tau_s", "s", "relaxation_time_constant"),
            ("rest_voltage_delta_v", "V", "relaxation_voltage_recovery"),
            ("rest_voltage_linear_slope_v_per_s", "V/s", "relaxation_slope"),
        ):
            first = _first_valid_row(ordered, column)
            last = _last_valid_row(ordered, column)
            if first is None or last is None:
                continue
            first_value = _number(first.get(column))
            last_value = _number(last.get(column))
            if first_value is None or last_value is None:
                continue
            first_cycle = _int_or_none(first.get("cycle_index"))
            last_cycle = _int_or_none(last.get("cycle_index"))
            records.append(
                _record(
                    cell_id=str(cell_id),
                    source_table="relaxation_features",
                    feature_name=f"{column}_change",
                    evidence_type="derived_feature",
                    value=last_value - first_value,
                    unit=unit,
                    cycle_start=first_cycle,
                    cycle_end=last_cycle,
                    support_role=role,
                    reliability="medium",
                    interpretation_hint=(
                        "Relaxation changes may support polarization, resistance-growth, or "
                        "transport hypotheses, but are protocol-dependent."
                    ),
                    text=(
                        f"{column} changed from {first_value:.4g} at cycle {first_cycle} to "
                        f"{last_value:.4g} at cycle {last_cycle}."
                    ),
                )
            )
    return records


def _stress_candidates(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        cell_id = str(row.get("cell_id"))
        for column, unit, role, hint in (
            (
                "high_soc_rest_fraction",
                "fraction",
                "usage_stress",
                "High-SOC rest exposure is a calendar-aging stressor for many Li-ion chemistries.",
            ),
            (
                "max_instant_discharge_c_rate",
                "C",
                "usage_stress",
                "High discharge C-rate can support load-stress and heating concerns when "
                "compared with a cell specification.",
            ),
            (
                "throughput_ah",
                "Ah",
                "usage_history",
                "Throughput summarizes cycling exposure and helps contextualize aging severity.",
            ),
            (
                "temperature_c_mean",
                "degC",
                "thermal_context",
                "Mean temperature provides thermal context but does not capture all thermal "
                "excursions.",
            ),
        ):
            value = _number(row.get(column))
            if value is None:
                continue
            records.append(
                _record(
                    cell_id=cell_id,
                    source_table="stress_features",
                    source_row_index=_row_index(row),
                    feature_name=column,
                    evidence_type="derived_feature",
                    value=value,
                    unit=unit,
                    support_role=role,
                    reliability="medium",
                    interpretation_hint=hint,
                    text=f"{column} is {value:.4g} {unit}.",
                )
            )
    return records


def _diagnostic_tag_candidates(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        signal = str(row.get("signal", "diagnostic_signal"))
        confidence = str(row.get("confidence", "low"))
        modes = row.get("possible_modes")
        if isinstance(modes, (list, tuple)):
            mode_text = ", ".join(str(mode) for mode in modes)
        else:
            mode_text = str(modes) if modes is not None else ""
        evidence = str(row.get("evidence", ""))
        records.append(
            _record(
                cell_id=str(row.get("cell_id")),
                source_table="degradation_tags",
                source_row_index=_row_index(row),
                feature_name=signal,
                evidence_type="diagnostic_tag",
                support_role="diagnostic_signal",
                reliability=_confidence_to_reliability(confidence),
                interpretation_hint=(
                    "Rule-based diagnostic tags are supporting evidence candidates. They should "
                    "not be treated as confirmed root-cause labels without corroborating features."
                ),
                text=(
                    f"Diagnostic signal {signal} fired with {confidence} confidence. "
                    f"{evidence} Possible modes: {mode_text}."
                ),
            )
        )
    return records


def _record(
    *,
    cell_id: str,
    source_table: str,
    feature_name: str,
    evidence_type: str,
    support_role: str,
    reliability: str,
    interpretation_hint: str,
    text: str,
    value: float | None = None,
    unit: str | None = None,
    source_row_index: int | None = None,
    cycle_index: int | None = None,
    cycle_start: int | None = None,
    cycle_end: int | None = None,
    protocol: str = "unknown",
) -> dict[str, Any]:
    evidence_id = _evidence_id(
        cell_id,
        source_table,
        feature_name,
        cycle_index,
        cycle_start,
        cycle_end,
    )
    redundancy_key = f"{cell_id}:{source_table}:{support_role}:{feature_name}"
    return {
        "cell_id": cell_id,
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "source_table": source_table,
        "source_row_index": source_row_index,
        "cycle_index": cycle_index,
        "cycle_start": cycle_start,
        "cycle_end": cycle_end,
        "feature_name": feature_name,
        "value": value,
        "unit": unit,
        "protocol": protocol,
        "support_role": support_role,
        "reliability": reliability,
        "interpretation_hint": interpretation_hint,
        "text": text,
        "token_cost": _estimate_token_cost(text, interpretation_hint),
        "redundancy_key": redundancy_key,
        "question_relevance": None,
        "protocol_consistency": None,
        "reliability_score": None,
        "score": None,
        "selection_rank": None,
        "selected": False,
    }


def _evidence_id(
    cell_id: str,
    source_table: str,
    feature_name: str,
    cycle_index: int | None,
    cycle_start: int | None,
    cycle_end: int | None,
) -> str:
    cycle = cycle_index if cycle_index is not None else f"{cycle_start or 'na'}_{cycle_end or 'na'}"
    raw = f"{cell_id}_{source_table}_{feature_name}_{cycle}"
    return "".join(ch if ch.isalnum() else "_" for ch in raw.lower()).strip("_")


def _estimate_token_cost(*parts: str) -> int:
    text = " ".join(part for part in parts if part)
    return max(8, int(np.ceil(len(text) / 4)))


def _sort_by_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    if "cycle_index" in frame.columns:
        return frame.sort_values("cycle_index")
    return frame


def _first_valid_row(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    subset = frame[pd.to_numeric(frame[column], errors="coerce").notna()]
    if subset.empty:
        return None
    return subset.iloc[0]


def _last_valid_row(frame: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in frame.columns:
        return None
    subset = frame[pd.to_numeric(frame[column], errors="coerce").notna()]
    if subset.empty:
        return None
    return subset.iloc[-1]


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _int_or_none(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _row_index(row: pd.Series) -> int | None:
    try:
        return int(row.name)
    except (TypeError, ValueError):
        return None


def _protocol_value(row: pd.Series) -> str:
    value = row.get("protocol")
    if value is None or pd.isna(value) or not str(value).strip():
        return "unknown"
    return str(value)


def _confidence_to_reliability(confidence: str) -> str:
    mapping = {"high": "high", "medium": "medium", "low": "low"}
    return mapping.get(confidence.lower(), "low")
