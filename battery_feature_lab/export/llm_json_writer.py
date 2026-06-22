"""LLM-ready JSONL export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_llm_context_records(
    tables: dict[str, pd.DataFrame],
    degradation_tags: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Build compact cell-level records for LLM analysis."""

    cells = set()
    for frame in tables.values():
        if frame is not None and "cell_id" in frame.columns:
            cells.update(frame["cell_id"].dropna().astype(str).unique())
    if degradation_tags is not None and "cell_id" in degradation_tags.columns:
        cells.update(degradation_tags["cell_id"].dropna().astype(str).unique())

    records: list[dict[str, Any]] = []
    for cell_id in sorted(cells):
        record: dict[str, Any] = {
            "cell_id": cell_id,
            "summary": {},
            "features": {},
            "diagnostic_evidence": [],
            "provenance": {"source": "battery-feature-lab"},
        }
        for name, frame in tables.items():
            if frame is None or frame.empty or "cell_id" not in frame.columns:
                continue
            subset = frame[frame["cell_id"].astype(str) == cell_id]
            if subset.empty:
                continue
            if "cycle_index" in subset.columns:
                record["features"][name] = _summarize_table(subset)
            else:
                record["features"][name] = _clean_record(subset.iloc[0].to_dict())
        if degradation_tags is not None and not degradation_tags.empty:
            tags = degradation_tags[degradation_tags["cell_id"].astype(str) == cell_id]
            record["diagnostic_evidence"] = [_clean_record(row) for row in tags.to_dict("records")]
        records.append(record)
    return records


def write_llm_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write LLM context records as JSON Lines."""

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


def _clean_record(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_record(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_record(v) for v in value]
    return _json_scalar(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value
