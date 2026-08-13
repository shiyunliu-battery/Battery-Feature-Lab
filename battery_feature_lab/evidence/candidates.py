"""Convert explicitly contracted feature outputs into evidence records."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from battery_feature_lab.evidence.schema import empty_evidence_frame, normalize_evidence_frame
from battery_feature_lab.feature_contracts import (
    FeatureContract,
    get_feature_contract,
    resolve_method_parameters,
)


def build_evidence_candidates(
    tables: dict[str, pd.DataFrame],
    degradation_tags: pd.DataFrame | None = None,
    *,
    contract_parameters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build controlled AI-visible evidence; unregistered numerics are never exposed."""

    records: list[dict[str, Any]] = []
    records.extend(_cycle_feature_candidates(tables.get("cycle_features"), contract_parameters))
    records.extend(_delta_q_candidates(tables.get("delta_q_features"), contract_parameters))
    records.extend(_ica_dva_candidates(tables.get("ica_dva_features"), contract_parameters))
    records.extend(_relaxation_candidates(tables.get("relaxation_features"), contract_parameters))
    records.extend(_stress_candidates(tables.get("stress_features"), contract_parameters))
    records.extend(_diagnostic_observation_candidates(degradation_tags, contract_parameters))
    records = _attach_protocol_metadata(records, tables.get("protocol_segments"))
    records = _attach_source_intervals(records, tables.get("normalized_timeseries"))
    records = _finalise_runtime_status(records)
    if not records:
        return empty_evidence_frame()
    return normalize_evidence_frame(pd.DataFrame.from_records(records))


def _cycle_feature_candidates(frame, contract_parameters):
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records = []
    for cell_id, group in frame.groupby("cell_id", sort=True):
        ordered = _sort_by_cycle(group)
        first = _first_valid_row(ordered, "discharge_capacity_ah")
        last = _last_valid_row(ordered, "discharge_capacity_ah")
        if first is not None and last is not None:
            first_cap = _number(first.get("discharge_capacity_ah"))
            last_cap = _number(last.get("discharge_capacity_ah"))
            if first_cap not in (None, 0.0) and last_cap is not None:
                records.append(_record(
                    cell_id=str(cell_id), source_table="cycle_features",
                    feature_name="discharge_capacity_retention", evidence_type="derived_feature",
                    value=last_cap / first_cap, unit="fraction",
                    cycle_start=_int_or_none(first.get("cycle_index")),
                    cycle_end=_int_or_none(last.get("cycle_index")), support_role="capacity_trend",
                    reliability="high",
                    interpretation_hint="Capacity retention supports fade/retention statements but not a unique mechanism.",
                    text=f"Discharge capacity changed from {first_cap:.6g} Ah at cycle {_int_or_none(first.get('cycle_index'))} to {last_cap:.6g} Ah at cycle {_int_or_none(last.get('cycle_index'))}.",
                    contract_parameters=contract_parameters,
                ))
        for label, row in (("first", _first_valid_row(ordered, "coulombic_efficiency")), ("last", _last_valid_row(ordered, "coulombic_efficiency"))):
            if row is None:
                continue
            value = _number(row.get("coulombic_efficiency"))
            if value is None:
                continue
            cycle = _int_or_none(row.get("cycle_index"))
            records.append(_record(
                cell_id=str(cell_id), source_table="cycle_features", source_row_index=_row_index(row),
                cycle_index=cycle, feature_name="coulombic_efficiency", evidence_type="derived_feature",
                value=value, unit="fraction", support_role="efficiency", reliability="medium",
                interpretation_hint="Coulombic efficiency is descriptive evidence for cycle consistency; it is not a mechanism label.",
                text=f"{label.capitalize()} coulombic efficiency is {value:.6g} at cycle {cycle}.",
                contract_parameters=contract_parameters,
            ))
    return records


def _delta_q_candidates(frame, contract_parameters):
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records = []
    for _, row in frame.iterrows():
        cell_id = str(row.get("cell_id")); ref = _int_or_none(row.get("reference_cycle")); target = _int_or_none(row.get("target_cycle"))
        for column, unit in (("delta_q_abs_area_ah_v", "Ah*V"), ("delta_q_variance", "Ah^2"), ("delta_q_l2", "Ah")):
            value = _number(row.get(column))
            if value is None:
                continue
            records.append(_record(
                cell_id=cell_id, source_table="delta_q_features", source_row_index=_row_index(row),
                feature_name=column, evidence_type="derived_feature", value=value, unit=unit,
                cycle_start=ref, cycle_end=target, support_role="delta_q_change", reliability="medium",
                interpretation_hint="Delta Q(V) is a curve-change descriptor; rate, temperature and protocol comparability must be considered.",
                text=f"{column} between cycles {ref} and {target} is {value:.6g} {unit}.",
                contract_parameters=contract_parameters, applicability_status="conditional",
            ))
    return records


def _ica_dva_candidates(frame, contract_parameters):
    """Expose area changes only; untracked peak identities are not AI-visible."""
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records = []
    group_columns = ["cell_id"] + (["step_type"] if "step_type" in frame.columns else [])
    for key, group in frame.groupby(group_columns, sort=True):
        cell_id, step_type = (str(key[0]), str(key[1])) if isinstance(key, tuple) else (str(key), "unknown")
        ordered = _sort_by_cycle(group)
        for column, unit, role in (("ica_area", "Ah", "ica_area"), ("dva_area", "V", "dva_area")):
            first = _first_valid_row(ordered, column); last = _last_valid_row(ordered, column)
            if first is None or last is None:
                continue
            first_value = _number(first.get(column)); last_value = _number(last.get(column))
            first_cycle = _int_or_none(first.get("cycle_index")); last_cycle = _int_or_none(last.get("cycle_index"))
            if first_value is None or last_value is None or first_cycle == last_cycle:
                continue
            records.append(_record(
                cell_id=cell_id, source_table="ica_dva_features", feature_name=f"{column}_change",
                evidence_type="derived_feature", value=last_value-first_value, unit=unit,
                cycle_start=first_cycle, cycle_end=last_cycle, support_role=role, reliability="medium",
                interpretation_hint=f"{column} change is compared within {step_type} segments only and is not a mechanism diagnosis.",
                text=f"{step_type} {column} changed from {first_value:.6g} at cycle {first_cycle} to {last_value:.6g} at cycle {last_cycle}.",
                contract_parameters=contract_parameters, applicability_status="conditional",
            ))
    return records


def _relaxation_candidates(frame, contract_parameters):
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records = []
    group_columns = ["cell_id"] + (["step_index"] if "step_index" in frame.columns else [])
    for key, group in frame.groupby(group_columns, sort=True):
        cell_id, step_index = (str(key[0]), _int_or_none(key[1])) if isinstance(key, tuple) else (str(key), None)
        ordered = _sort_by_cycle(group)
        for column, unit, role in (("rest_exp_tau_s", "s", "relaxation_time_constant"), ("rest_voltage_delta_v", "V", "relaxation_voltage_recovery"), ("rest_voltage_linear_slope_v_per_s", "V/s", "relaxation_slope")):
            first = _first_valid_row(ordered, column); last = _last_valid_row(ordered, column)
            if first is None or last is None:
                continue
            first_value = _number(first.get(column)); last_value = _number(last.get(column))
            first_cycle = _int_or_none(first.get("cycle_index")); last_cycle = _int_or_none(last.get("cycle_index"))
            if first_value is None or last_value is None or first_cycle is None or first_cycle == last_cycle:
                continue
            records.append(_record(
                cell_id=cell_id, source_table="relaxation_features", feature_name=f"{column}_change",
                evidence_type="derived_feature", value=last_value-first_value, unit=unit,
                cycle_start=first_cycle, cycle_end=last_cycle, step_index=step_index,
                support_role=role, reliability="medium",
                interpretation_hint="Relaxation endpoint comparison is matched by step index; comparable pre-rest state and duration must still be checked.",
                text=f"{column} at step {step_index} changed from {first_value:.6g} at cycle {first_cycle} to {last_value:.6g} at cycle {last_cycle}.",
                contract_parameters=contract_parameters, applicability_status="conditional",
            ))
    return records


def _stress_candidates(frame, contract_parameters):
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records = []
    for _, row in frame.iterrows():
        candidates = [("max_instant_discharge_c_rate", "C", "usage_stress"), ("throughput_ah", "Ah", "usage_history"), ("temperature_c_mean", "degC", "thermal_context")]
        if str(row.get("soc_source", "")).lower() == "provided":
            candidates.insert(0, ("high_soc_rest_fraction", "fraction", "usage_stress"))
        for column, unit, role in candidates:
            value = _number(row.get(column))
            if value is None:
                continue
            records.append(_record(
                cell_id=str(row.get("cell_id")), source_table="stress_features", source_row_index=_row_index(row),
                feature_name=column, evidence_type="derived_feature", value=value, unit=unit,
                support_role=role, reliability="medium",
                interpretation_hint="Usage and environmental descriptors provide context only; they do not establish a degradation mechanism.",
                text=f"{column} is {value:.6g} {unit}.", contract_parameters=contract_parameters,
            ))
    return records


def _diagnostic_observation_candidates(frame, contract_parameters):
    if frame is None or frame.empty or "cell_id" not in frame.columns:
        return []
    records = []
    for _, row in frame.iterrows():
        signal = str(row.get("signal", "")); contract = get_feature_contract("degradation_tags", signal)
        if contract is None or not contract.ai_visible:
            continue
        evidence = str(row.get("evidence", ""))
        records.append(_record(
            cell_id=str(row.get("cell_id")), source_table="degradation_tags", source_row_index=_row_index(row),
            cycle_start=_int_or_none(row.get("cycle_start")), cycle_end=_int_or_none(row.get("cycle_end")),
            feature_name=signal, evidence_type="observation", support_role="observed_signal", reliability="medium",
            interpretation_hint="This is an observation or stress flag; it must not be promoted to LLI, LAM or another mechanism without corroboration.",
            text=evidence or f"Observation {signal} was detected.", contract_parameters=contract_parameters,
        ))
    return records


def _record(*, cell_id, source_table, feature_name, evidence_type, support_role, reliability, interpretation_hint, text, value=None, unit=None, source_row_index=None, cycle_index=None, cycle_start=None, cycle_end=None, step_index=None, protocol="unknown", contract_parameters=None, applicability_status="applicable"):
    contract = _require_ai_visible_contract(source_table, feature_name)
    parameters = resolve_method_parameters(contract, contract_parameters)
    evidence_id = _evidence_id(cell_id, source_table, feature_name, cycle_index, cycle_start, cycle_end, step_index)
    return {
        "cell_id": cell_id, "evidence_id": evidence_id, "evidence_type": evidence_type,
        "source_table": source_table, "source_row_index": source_row_index, "cycle_index": cycle_index,
        "cycle_start": cycle_start, "cycle_end": cycle_end, "step_index": step_index,
        "source_time_start_s": None, "source_time_end_s": None, "feature_name": feature_name,
        "value": value, "unit": unit or contract.unit, "contract_id": contract.contract_id,
        "definition": contract.definition, "inputs": json.dumps(list(contract.inputs), separators=(",", ":")),
        "method": contract.method, "method_version": contract.method_version,
        "method_parameters": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
        "applicability": contract.applicability, "applicability_status": applicability_status,
        "quality": contract.quality, "quality_status": "pass", "source_interval": contract.source_interval,
        "interpretation_level": contract.interpretation_level, "protocol": protocol,
        "support_role": support_role, "reliability": reliability, "interpretation_hint": interpretation_hint,
        "text": text, "token_cost": _estimate_token_cost(text, interpretation_hint, contract.definition),
        "redundancy_key": f"{cell_id}:{source_table}:{support_role}:{feature_name}:{step_index}",
        "question_relevance": None, "protocol_consistency": None, "reliability_score": None,
        "score": None, "selection_rank": None, "selected": False,
    }


def _require_ai_visible_contract(source_table, feature_name):
    contract = get_feature_contract(source_table, feature_name)
    if contract is None:
        raise ValueError(f"Feature {source_table}.{feature_name} has no Feature Contract and cannot enter evidence.")
    if not contract.ai_visible:
        raise ValueError(f"Feature {source_table}.{feature_name} is withheld from AI-visible evidence.")
    return contract


def _attach_protocol_metadata(records, protocol_segments):
    if not records or protocol_segments is None or protocol_segments.empty:
        return records
    if not {"cell_id", "cycle_index", "protocol"}.issubset(protocol_segments.columns):
        return records
    lookup = {}
    for (cell_id, cycle_index), group in protocol_segments.groupby(["cell_id", "cycle_index"], sort=False):
        ordered = group.sort_values("segment_index") if "segment_index" in group.columns else group
        first = ordered.iloc[0]
        lookup[(str(cell_id), int(cycle_index))] = {"protocol": _text_or_unknown(first.get("protocol")), "protocol_family": _text_or_unknown(first.get("protocol_family")), "protocol_confidence": _number(first.get("protocol_confidence"))}
    for record in records:
        contexts = _contexts_for_record(record, lookup)
        if not contexts:
            continue
        protocols = {c["protocol"] for c in contexts}; families = {c["protocol_family"] for c in contexts}; confidences = [c["protocol_confidence"] for c in contexts if c["protocol_confidence"] is not None]
        record["protocol"] = protocols.pop() if len(protocols) == 1 else "mixed"
        record["protocol_family"] = families.pop() if len(families) == 1 else "mixed"
        record["protocol_confidence"] = min(confidences) if confidences else None
        if record["protocol"] == "mixed" and record["applicability_status"] == "applicable":
            record["applicability_status"] = "conditional"
    return records


def _contexts_for_record(record, lookup):
    cell_id = str(record.get("cell_id")); cycle_index = _int_or_none(record.get("cycle_index")); cycle_start = _int_or_none(record.get("cycle_start")); cycle_end = _int_or_none(record.get("cycle_end"))
    if cycle_index is not None:
        context = lookup.get((cell_id, cycle_index)); return [context] if context is not None else []
    if cycle_start is not None and cycle_end is not None:
        return [context for (candidate_cell, candidate_cycle), context in lookup.items() if candidate_cell == cell_id and cycle_start <= candidate_cycle <= cycle_end]
    return []


def _attach_source_intervals(records, normalized):
    if not records or normalized is None or normalized.empty or not {"cell_id", "time_s"}.issubset(normalized.columns):
        return records
    for record in records:
        subset = normalized[normalized["cell_id"].astype(str) == str(record["cell_id"])]
        cycle_index = _int_or_none(record.get("cycle_index")); cycle_start = _int_or_none(record.get("cycle_start")); cycle_end = _int_or_none(record.get("cycle_end")); step_index = _int_or_none(record.get("step_index"))
        if "cycle_index" in subset.columns:
            cycles = pd.to_numeric(subset["cycle_index"], errors="coerce")
            if cycle_index is not None:
                subset = subset[cycles == cycle_index]
            elif cycle_start is not None and cycle_end is not None:
                subset = subset[(cycles >= cycle_start) & (cycles <= cycle_end)]
        if step_index is not None and "step_index" in subset.columns:
            subset = subset[pd.to_numeric(subset["step_index"], errors="coerce") == step_index]
        time = pd.to_numeric(subset["time_s"], errors="coerce").dropna()
        if not time.empty:
            record["source_time_start_s"] = float(time.min()); record["source_time_end_s"] = float(time.max())
    return records


def _finalise_runtime_status(records):
    for record in records:
        if record.get("source_time_start_s") is None or record.get("source_time_end_s") is None or record.get("applicability_status") == "conditional":
            record["quality_status"] = "warning"
    return records


def _evidence_id(cell_id, source_table, feature_name, cycle_index, cycle_start, cycle_end, step_index):
    cycle = cycle_index if cycle_index is not None else f"{cycle_start or 'na'}_{cycle_end or 'na'}"
    raw = f"{cell_id}_{source_table}_{feature_name}_{cycle}_{step_index if step_index is not None else 'na'}"
    return "".join(ch if ch.isalnum() else "_" for ch in raw.lower()).strip("_")


def _estimate_token_cost(*parts):
    text = " ".join(part for part in parts if part)
    return max(8, int(np.ceil(len(text) / 4)))


def _sort_by_cycle(frame):
    return frame.sort_values("cycle_index") if "cycle_index" in frame.columns else frame


def _first_valid_row(frame, column):
    if column not in frame.columns:
        return None
    subset = frame[pd.to_numeric(frame[column], errors="coerce").notna()]
    return None if subset.empty else subset.iloc[0]


def _last_valid_row(frame, column):
    if column not in frame.columns:
        return None
    subset = frame[pd.to_numeric(frame[column], errors="coerce").notna()]
    return None if subset.empty else subset.iloc[-1]


def _number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _int_or_none(value):
    number = _number(value)
    return int(number) if number is not None else None


def _row_index(row):
    try:
        return int(row.name)
    except (TypeError, ValueError):
        return None


def _text_or_unknown(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return "unknown"
    return str(value)
