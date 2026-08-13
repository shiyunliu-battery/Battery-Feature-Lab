"""Build LLM context exclusively from contracted evidence records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_feature_lab.feature_contracts import FEATURE_CONTRACTS, contract_to_dict

_CONTRACTS_BY_ID = {contract.contract_id: contract for contract in FEATURE_CONTRACTS.values()}


def build_llm_context_records(evidence: pd.DataFrame | None, *, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build one LLM context record per cell from evidence only."""
    if evidence is None or evidence.empty or "cell_id" not in evidence.columns:
        return []
    metadata = metadata or {}; records = []
    for cell_id, group in evidence.groupby("cell_id", sort=True):
        ordered = _order_evidence(group)
        evidence_records = [_clean_record(row) for row in ordered.to_dict("records")]
        contract_ids = [str(value) for value in ordered.get("contract_id", pd.Series(dtype=str)).dropna().unique() if str(value) in _CONTRACTS_BY_ID]
        contracts = {contract_id: contract_to_dict(_CONTRACTS_BY_ID[contract_id]) for contract_id in sorted(contract_ids)}
        warnings = [item["evidence_id"] for item in evidence_records if item.get("quality_status") == "warning" or item.get("applicability_status") in {"conditional", "not_assessed"}]
        records.append(_clean_record({
            "schema_version": "3.0", "cell_id": str(cell_id),
            "evidence_chain": {"policy": "feature -> Feature Contract -> evidence record -> LLM context", "feature_table_bypass": False, "contract_required": True},
            "cell_context": metadata.get("cell_context", {}), "analysis_config": metadata.get("analysis_config", {}),
            "evidence_count": len(evidence_records),
            "status_summary": {"quality": _value_counts(ordered, "quality_status"), "applicability": _value_counts(ordered, "applicability_status"), "interpretation_level": _value_counts(ordered, "interpretation_level")},
            "summary": {"evidence_ids": [item.get("evidence_id") for item in evidence_records], "statements": [item.get("text") for item in evidence_records if item.get("text")]},
            "evidence": evidence_records, "contracts": contracts,
            "review_notes": {"warning_evidence_ids": warnings, "interpretation_boundary": "Observation and derived features may support review. No evidence record is an automatic LLI/LAM or root-cause diagnosis."},
            "provenance": {"source": "battery-feature-lab", "bds_conversion_report": metadata.get("bds_conversion_report", {})},
        }))
    return records


def write_llm_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def _order_evidence(frame):
    if "selection_rank" in frame.columns and frame["selection_rank"].notna().any():
        return frame.sort_values(["selection_rank", "evidence_id"], na_position="last")
    return frame.sort_values("evidence_id") if "evidence_id" in frame.columns else frame


def _value_counts(frame, column):
    if column not in frame.columns:
        return {}
    values = frame[column].dropna().astype(str)
    return {key: int(value) for key, value in values.value_counts().sort_index().items()}


def _clean_record(value):
    if isinstance(value, dict):
        return {str(key): _clean_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_record(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_clean_record(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
