"""JSONL writer for evidence records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_evidence_jsonl(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write evidence records as JSON Lines."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        if frame is not None and not frame.empty:
            for record in frame.to_dict("records"):
                handle.write(json.dumps(_clean_record(record), ensure_ascii=False) + "\n")
    return output


def _clean_record(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_record(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_clean_record(v) for v in value.tolist()]
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
