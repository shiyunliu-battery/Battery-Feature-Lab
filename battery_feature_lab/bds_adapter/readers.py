"""Ingest layer for BFL.

File parsing, cycler detection, column mapping, unit conversion, time-axis repair and
current-sign resolution are delegated to :mod:`bds` (the ``battery-data-standard``
package). BFL does not reimplement any of that.

What stays here is the small semantic layer BFL needs on top of the BDS table and that
BDS deliberately does not model: ``cell_id``, ``step_type``, a ``step_index`` fallback,
and recovery of pass-through columns such as ``soc`` and the EIS triple.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import bds
import pandas as pd

from battery_feature_lab.bds_adapter.validators import validate_timeseries
from battery_feature_lab.core.cycle_splitter import infer_cycle_index, infer_step_type
from battery_feature_lab.schemas import ReaderConfig

# BDS canonical label -> BFL canonical label. Anything not listed keeps the BDS name.
_BDS_TO_BFL = {
    "test_time_s": "time_s",
    "ambient_temperature_deg_c": "temperature_c",
    "temperature_t1_deg_c": "temperature_c",
    "charge_energy_wh": "charge_energy_wh",
    "discharge_energy_wh": "discharge_energy_wh",
}

# BDS required schema. Absence means the file was not recognised as cycler data.
_BDS_REQUIRED = ("test_time_s", "voltage_v", "current_a")

# Formats BDS does not read natively. These are materialised to CSV and handed to BDS
# rather than parsed by BFL, so there is still exactly one normalization path.
_JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}


def read_bds_export(path: str | Path, config: ReaderConfig | None = None) -> pd.DataFrame:
    """Read a cycler export into BFL canonical column names via BDS.

    Parameters are unchanged from previous releases. Supported inputs are whatever the
    installed BDS version supports, plus JSON and JSONL.
    """

    frame, _ = read_bds_export_with_report(path, config)
    return frame


def read_bds_export_with_report(
    path: str | Path, config: ReaderConfig | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Same as :func:`read_bds_export` but also returns the BDS conversion report.

    The report carries adapter identity, unit transforms, repairs and warnings. Keep it
    with the feature tables so the provenance of a run stays auditable.
    """

    config = config or ReaderConfig()
    path = Path(path)

    if config.positive_current_is_charge is False:
        warnings.warn(
            "ReaderConfig.positive_current_is_charge is ignored. BDS resolves the source "
            "current-sign convention from the adapter and normalizes to charge-positive.",
            DeprecationWarning,
            stacklevel=3,
        )

    raw_frame, report = _read_via_bds(path)
    normalized = _to_bfl_columns(raw_frame, config)
    normalized = _apply_bfl_semantics(normalized, config, default_cell_id=path.stem)
    validate_timeseries(normalized)
    return normalized, report


def _read_via_bds(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Call BDS and return a pandas frame plus a serializable conversion report."""

    if path.suffix.lower() in _JSON_SUFFIXES:
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / f"{path.stem}.csv"
            _stage_json_as_csv(path, staged)
            return _read_via_bds(staged)

    try:
        frame, report = bds.read_with_report(
            path,
            cycler="auto",
            strict=False,
            keep_raw=True,
            current_sign="charge-positive",
            repair_policy="repair",
        )
    except bds.BatteryDataStandardError as exc:
        raise ValueError(
            f"BDS could not read {path.name}: {exc}. "
            f"Run `bds doctor {path}` for adapter candidates and missing required columns."
        ) from exc

    missing = [c for c in _BDS_REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(
            f"BDS read {path.name} but could not identify the required columns {missing}. "
            f"Run `bds doctor {path}` for adapter candidates and suggested next steps, or pass "
            f"a column profile via `bds convert --profile`."
        )

    return frame.to_pandas(), _report_to_dict(report)


def _stage_json_as_csv(source: Path, target: Path) -> None:
    """Materialise JSON/JSONL as CSV so BDS owns the actual normalization."""

    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        records = pd.read_json(source, lines=True)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("data") or payload.get("records") or payload.get("rows") or [payload]
        else:
            rows = payload
        records = pd.DataFrame.from_records(rows)
    records.to_csv(target, index=False)


def _report_to_dict(report: Any) -> dict[str, Any]:
    for attr in ("to_dict", "as_dict", "model_dump"):
        method = getattr(report, attr, None)
        if callable(method):
            try:
                return dict(method())
            except TypeError:
                continue
    if hasattr(report, "__dict__"):
        return {k: v for k, v in vars(report).items() if not k.startswith("_")}
    return {"report": str(report)}


def _to_bfl_columns(frame: pd.DataFrame, config: ReaderConfig) -> pd.DataFrame:
    """Rename BDS canonical labels to BFL labels and recover pass-through columns.

    BDS returns unmapped source columns under a ``raw:`` prefix. Those are matched
    against the BFL alias table so that columns BDS does not model, notably ``soc``,
    ``cell_id``, ``step_type`` and the EIS triple, survive the handoff.
    """

    canonical = {c: _BDS_TO_BFL.get(c, c) for c in frame.columns if not c.startswith("raw:")}
    normalized = frame[list(canonical)].rename(columns=canonical).copy()

    passthrough = {c: c.split("raw:", 1)[1] for c in frame.columns if c.startswith("raw:")}
    if not passthrough:
        return normalized

    lookup = {_clean_name(original): raw for raw, original in passthrough.items()}
    for target, aliases in config.column_aliases.aliases.items():
        if target in normalized.columns:
            continue
        for candidate in (target, *aliases):
            source = lookup.get(_clean_name(candidate))
            if source is None:
                continue
            values = frame[source]
            if target in {"cell_id", "step_type"}:
                normalized[target] = values
            else:
                normalized[target] = pd.to_numeric(values, errors="coerce")
            break

    if "soc" in normalized.columns and config.soc_unit.lower() in {"percent", "%"}:
        normalized["soc"] = normalized["soc"] / 100.0

    return normalized


def _apply_bfl_semantics(
    frame: pd.DataFrame, config: ReaderConfig, default_cell_id: str
) -> pd.DataFrame:
    """Add the BFL-only fields on top of the BDS table."""

    normalized = frame.copy()

    if "cell_id" not in normalized.columns:
        normalized["cell_id"] = config.cell_id or default_cell_id
    elif config.cell_id is not None:
        normalized["cell_id"] = normalized["cell_id"].fillna(config.cell_id)

    if "cycle_index" not in normalized.columns:
        normalized["cycle_index"] = infer_cycle_index(normalized)
    else:
        normalized["cycle_index"] = (
            pd.to_numeric(normalized["cycle_index"], errors="coerce").ffill().fillna(1).astype(int)
        )

    if "step_type" not in normalized.columns:
        normalized["step_type"] = infer_step_type(
            normalized["current_a"], config.current_rest_threshold_a
        )
    else:
        normalized["step_type"] = normalized["step_type"].map(_normalize_step_type)
        missing = normalized["step_type"].isna()
        if missing.any():
            normalized.loc[missing, "step_type"] = infer_step_type(
                normalized.loc[missing, "current_a"], config.current_rest_threshold_a
            )

    if "step_index" not in normalized.columns:
        normalized["step_index"] = _infer_step_index(normalized)

    normalized = _split_generic_capacity(normalized)
    return normalized.sort_values(["cell_id", "cycle_index", "time_s"]).reset_index(drop=True)


def _clean_name(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _normalize_step_type(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if any(token in text for token in ("discharge", "dchg", "dischg")):
        return "discharge"
    if any(token in text for token in ("charge", "chg", "cccv")):
        return "charge"
    if any(token in text for token in ("rest", "pause", "ocv", "relax")):
        return "rest"
    return None


def _split_generic_capacity(frame: pd.DataFrame) -> pd.DataFrame:
    if "capacity_ah" not in frame.columns:
        return frame
    normalized = frame.copy()
    capacity = pd.to_numeric(normalized["capacity_ah"], errors="coerce")
    if "charge_capacity_ah" not in normalized.columns:
        normalized["charge_capacity_ah"] = capacity.where(normalized["step_type"] == "charge", 0.0)
    if "discharge_capacity_ah" not in normalized.columns:
        normalized["discharge_capacity_ah"] = capacity.where(
            normalized["step_type"] == "discharge", 0.0
        )
    return normalized


def _infer_step_index(frame: pd.DataFrame) -> pd.Series:
    indices: list[int] = []
    current_step = 0
    previous_key: tuple[object, object, object] | None = None
    for row in frame[["cell_id", "cycle_index", "step_type"]].itertuples(index=False, name=None):
        if previous_key is None or row != previous_key:
            if previous_key is None or row[:2] != previous_key[:2]:
                current_step = 0
            else:
                current_step += 1
        indices.append(current_step)
        previous_key = row
    return pd.Series(indices, index=frame.index, dtype="int64")
