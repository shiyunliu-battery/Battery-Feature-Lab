"""Readers for BDS-style exports and common cycler tabular files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from battery_feature_lab.bds_adapter.validators import validate_timeseries
from battery_feature_lab.core.cycle_splitter import infer_cycle_index, infer_step_type
from battery_feature_lab.core.units import normalize_units_and_sign
from battery_feature_lab.schemas import ReaderConfig


def read_bds_export(path: str | Path, config: ReaderConfig | None = None) -> pd.DataFrame:
    """Read a BDS-style export into canonical column names.

    Supported input formats are CSV, TSV, JSON, JSONL and Parquet. JSON files may contain
    either a list of records or a dictionary with a top-level ``data`` records list.
    """

    config = config or ReaderConfig()
    path = Path(path)
    raw = _read_table(path)
    normalized = normalize_columns(raw, config)
    normalized = normalize_units_and_sign(normalized, config)

    if "cell_id" not in normalized.columns:
        normalized["cell_id"] = config.cell_id or path.stem
    elif config.cell_id is not None:
        normalized["cell_id"] = normalized["cell_id"].fillna(config.cell_id)

    if "cycle_index" not in normalized.columns:
        normalized["cycle_index"] = infer_cycle_index(normalized)

    if "step_type" not in normalized.columns:
        normalized["step_type"] = infer_step_type(
            normalized["current_a"], config.current_rest_threshold_a
        )
    else:
        normalized["step_type"] = normalized["step_type"].map(_normalize_step_type)
        missing = normalized["step_type"].isna()
        if missing.any():
            inferred = infer_step_type(normalized.loc[missing, "current_a"], config.current_rest_threshold_a)
            normalized.loc[missing, "step_type"] = inferred

    if "step_index" not in normalized.columns:
        normalized["step_index"] = _infer_step_index(normalized)

    normalized = normalized.sort_values(["cell_id", "cycle_index", "time_s"]).reset_index(drop=True)
    validate_timeseries(normalized)
    return normalized


def normalize_columns(frame: pd.DataFrame, config: ReaderConfig) -> pd.DataFrame:
    """Rename known aliases to canonical names and preserve extra columns."""

    lookup = {_clean_name(col): col for col in frame.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in config.column_aliases.aliases.items():
        candidates = (canonical, *aliases)
        for candidate in candidates:
            original = lookup.get(_clean_name(candidate))
            if original is not None:
                rename[original] = canonical
                break

    normalized = frame.rename(columns=rename).copy()
    for column in ("time_s", "voltage_v", "current_a", "temperature_c"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in ("charge_capacity_ah", "discharge_capacity_ah", "energy_wh", "soc"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in ("cycle_index", "step_index"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int64")
    return normalized


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            records = payload.get("data") or payload.get("records") or payload.get("rows")
            if records is None:
                records = [payload]
        else:
            records = payload
        return pd.DataFrame.from_records(records)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _clean_name(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_step_type(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if any(token in text for token in ("charge", "chg", "cccv")):
        return "charge"
    if any(token in text for token in ("discharge", "dchg", "dischg")):
        return "discharge"
    if any(token in text for token in ("rest", "pause", "ocv", "relax")):
        return "rest"
    return None


def _infer_step_index(frame: pd.DataFrame) -> pd.Series:
    indices: list[int] = []
    current_step = 0
    previous_key: tuple[object, object, object] | None = None
    for row in frame[["cell_id", "cycle_index", "step_type"]].itertuples(index=False, name=None):
        key = row
        if previous_key is None or key != previous_key:
            if previous_key is None or key[:2] != previous_key[:2]:
                current_step = 0
            else:
                current_step += 1
        indices.append(current_step)
        previous_key = key
    return pd.Series(indices, index=frame.index, dtype="int64")
