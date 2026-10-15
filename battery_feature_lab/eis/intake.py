"""BDS-backed EIS intake with explicit spectrum identity and row provenance."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import polars as pl
from battery_data_standard.eis import normalize_eis_frame, validate_eis
from battery_data_standard.io import read_table_with_metadata

from battery_feature_lab.analysis.writer import canonical_sha256
from battery_feature_lab.bds_provider import bds_distribution
from battery_feature_lab.eis.schema import CONDITION_UNITS, EISConfig

ALIASES = {
    "frequency": ("Frequency_Hz", "Frequency (Hz)", "freq/Hz", "Frequency/Hz"),
    "real": ("Zre_exp_Ohm", "Zreal_Ohm", "Real_Ohm", "Re(Z)/Ohm", "R(ohm)"),
    "imaginary": (
        "Zim_exp_Ohm",
        "Zimag_Ohm",
        "Imag_Ohm",
        "Im(Z)/Ohm",
        "-Im(Z)/Ohm",
        "-Zim_exp_Ohm",
        "X(ohm)",
    ),
    "magnitude": ("Zmod_Ohm", "Zmod", "|Z|/Ohm", "Magnitude_Ohm"),
    "phase": (
        "Phase_exp_deg",
        "Phase (deg)",
        "Phase_deg",
        "Phase_rad",
        "Zphase_deg",
        "Phase",
        "Zphz",
    ),
}
CONDITION_COLUMNS = {
    "cell_id": ("Cell_ID", "Cell ID"),
    "temperature_deg_c": ("Temperature_degC", "Temperature (degC)", "Temperature_C"),
    "soc_fraction": ("SOC_fraction", "SOC / 1"),
    "rest_time_s": ("Rest_Time_s",),
    "ac_amplitude_v": ("AC_Amplitude_V",),
    "ac_amplitude_a": ("AC_Amplitude_A",),
    "dc_voltage_v": ("DC_Voltage_V",),
    "instrument": ("Instrument",),
    "protocol_id": ("Protocol_ID",),
    "cycle_index": ("Cycle_Index",),
    "sequence_index": ("Sequence_Index",),
    "elapsed_time_s": ("Elapsed_Time_s",),
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", name.lower())


def pick(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    matches = [c for c in columns if _slug(c) in {_slug(a) for a in aliases}]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous EIS columns: {matches}; supply an explicit mapping")
    return matches[0] if matches else None


def _number(value: Any) -> float:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return float("nan")


def _conditions(group: pl.DataFrame, declared: dict[str, Any]) -> dict[str, Any]:
    if set(declared) - set(CONDITION_UNITS):
        raise ValueError(
            f"Unknown declared EIS conditions: {sorted(set(declared) - set(CONDITION_UNITS))}"
        )
    result = {}
    for name, unit in CONDITION_UNITS.items():
        column = pick(group.columns, CONDITION_COLUMNS[name])
        values = (
            [
                v
                for v in group[column].drop_nulls().unique().to_list()
                if not isinstance(v, str) or v.strip()
            ]
            if column
            else []
        )
        if len(values) > 1:
            raise ValueError(f"Condition {name} varies within a spectrum; split the measurements")
        source_value = values[0] if values else None
        value = declared.get(name, source_value)
        if (
            name in declared
            and source_value is not None
            and str(value) != str(source_value)
            and (unit is None or _number(value) != _number(source_value))
        ):
            raise ValueError(f"Declared {name} conflicts with the source column")
        if value is not None and unit is not None:
            value = _number(value)
            if not math.isfinite(value):
                raise ValueError(f"Condition {name} must be finite")
            if name == "soc_fraction" and not 0 <= value <= 1:
                raise ValueError("soc_fraction must be between zero and one")
            if (
                name in {"rest_time_s", "ac_amplitude_v", "ac_amplitude_a", "elapsed_time_s"}
                and value < 0
            ):
                raise ValueError(f"Condition {name} must be non-negative")
        if value is not None and unit is None:
            value = str(value)
            if not value.strip():
                raise ValueError(f"Declared {name} must not be blank")
        result[name] = {
            "value": value,
            "unit": unit,
            "status": "declared"
            if name in declared
            else "reported"
            if value is not None
            else "not_reported",
            "source": "metadata" if name in declared else column,
        }
    return result


def read_spectra(source, config: EISConfig):
    bds_distribution()
    read = read_table_with_metadata(source, options={"sheet": config.sheet, "excel_sheets": "eis"})
    raw = read.data
    if "measurement_source_row" in raw.columns:
        raise ValueError("Input uses reserved column measurement_source_row")
    raw = raw.with_row_index("measurement_source_row")
    if not raw.height:
        raise ValueError("EIS source has no measurement rows")
    spectrum_column = pick(raw.columns, ("Spectrum_ID", "Measurement_ID", "Sweep_ID"))
    cell_column = pick(raw.columns, CONDITION_COLUMNS["cell_id"])
    group_columns = list(config.group_columns)
    if not group_columns and spectrum_column:
        group_columns = [c for c in (cell_column, spectrum_column) if c]
    if not group_columns and not config.single_spectrum:
        raise ValueError(
            "Spectrum identity is absent; supply group_columns or declare single_spectrum=True"
        )
    for name in group_columns:
        if (
            name not in raw.columns
            or raw[name].null_count()
            or any(isinstance(v, str) and not v.strip() for v in raw[name])
        ):
            raise ValueError(f"Missing or incomplete spectrum grouping column: {name}")
    groups = raw.partition_by(group_columns, maintain_order=True) if group_columns else [raw]
    if config.single_spectrum and len(groups) != 1:
        raise ValueError("single_spectrum declaration conflicts with multiple source spectra")
    column_map = {
        name: config.columns.get(name) or pick(raw.columns, aliases)
        for name, aliases in ALIASES.items()
    }
    if any(c not in raw.columns for c in config.columns.values()):
        raise ValueError("Explicit EIS column mapping names an absent column")
    for name in ("frequency", "real", "imaginary", "magnitude"):
        header = _slug(column_map[name] or "")
        if any(token in header for token in ("khz", "mhz", "mohm", "kohm")):
            raise ValueError(
                "EIS column mappings require Hz and Ohm; convert non-canonical units explicitly"
            )
    if column_map["frequency"] is None:
        raise ValueError("Declare the frequency column in Hz")
    polar = column_map["real"] is None or column_map["imaginary"] is None
    phase_unit = config.phase_unit
    if polar:
        if not column_map["magnitude"] or not column_map["phase"]:
            raise ValueError("EIS needs real/imaginary columns or magnitude/phase columns")
        phase_name = column_map["phase"].lower()
        header_unit = "deg" if "deg" in phase_name else "rad" if "rad" in phase_name else None
        if phase_unit and header_unit and phase_unit != header_unit:
            raise ValueError("Declared phase unit conflicts with the source header")
        phase_unit = phase_unit or header_unit
        if phase_unit is None:
            raise ValueError(
                "Magnitude/phase data require an explicit phase_unit; values are never used to guess units"
            )
    ids = [
        str(g[spectrum_column][0])
        if spectrum_column
        else "single"
        if not group_columns
        else "spectrum-" + canonical_sha256({c: g[c][0] for c in group_columns})[:12]
        for g in groups
    ]
    id_counts = Counter(ids)
    uids = [
        canonical_sha256({"spectrum_id": name, "group": {c: g[c][0] for c in group_columns}})[:16]
        for g, name in zip(groups, ids, strict=True)
    ]
    overrides = config.metadata.get("spectra", {})
    if set(overrides) - set(ids) - set(uids):
        raise ValueError("Metadata names a spectrum absent from the source")
    if any(not isinstance(value, dict) for value in overrides.values()):
        raise TypeError("Per-spectrum metadata must be an object")
    spectra, frames, audits = [], [], []
    for group, spectrum_id, uid in zip(groups, ids, uids, strict=True):
        declared = dict(config.metadata.get("defaults", {}))
        if spectrum_id in overrides and id_counts[spectrum_id] > 1:
            raise ValueError("Metadata spectrum ID is ambiguous across cells; use spectrum_uid")
        declared.update(overrides.get(uid, overrides.get(spectrum_id, {})))
        conditions = _conditions(group, declared)
        f = np.array([_number(x) for x in group[column_map["frequency"]]])
        if polar:
            magnitude = np.array([_number(x) for x in group[column_map["magnitude"]]])
            angle = np.array([_number(x) for x in group[column_map["phase"]]])
            angle = np.deg2rad(angle) if phase_unit == "deg" else angle
            real, imag = magnitude * np.cos(angle), magnitude * np.sin(angle)
            bad_magnitude = magnitude < 0
        else:
            real = np.array([_number(x) for x in group[column_map["real"]]])
            imag = np.array([_number(x) for x in group[column_map["imaginary"]]])
            if column_map["imaginary"].strip().startswith("-"):
                imag = -imag
            bad_magnitude = np.zeros(len(f), dtype=bool)
        valid = np.isfinite(f) & (f > 0) & np.isfinite(real) & np.isfinite(imag) & ~bad_magnitude
        staged = group.with_columns(
            pl.Series("Frequency_Hz", f),
            pl.Series("Zre_exp_Ohm", real),
            pl.Series("Zim_exp_Ohm", imag),
        ).filter(pl.Series(valid))
        if staged.height:
            normalized = normalize_eis_frame(staged)
            validation = validate_eis(normalized).to_dict()
        else:
            normalized = pl.DataFrame(
                schema={
                    "Frequency_Hz": pl.Float64,
                    "Zre_exp_Ohm": pl.Float64,
                    "Zim_exp_Ohm": pl.Float64,
                    "measurement_source_row": pl.UInt32,
                }
            )
            validation = {"valid": False, "reason": "no_valid_measurement_points"}
        retained = set(normalized["measurement_source_row"].to_list())
        dropped = [
            {
                "source_row": int(row),
                "reason": "negative_magnitude"
                if bad_magnitude[i]
                else "non_positive_frequency"
                if math.isfinite(f[i]) and f[i] <= 0
                else "invalid_numeric_point",
            }
            for i, row in enumerate(group["measurement_source_row"])
            if row not in retained
        ]
        if len(retained) + len(dropped) != group.height:
            raise RuntimeError("EIS row accounting failed")
        normalized = normalized.select(
            "Frequency_Hz", "Zre_exp_Ohm", "Zim_exp_Ohm", "measurement_source_row"
        )
        normalized = normalized.with_columns(pl.lit(uid).alias("spectrum_uid"))
        frames.append(normalized)
        spectra.append(
            {
                "spectrum_uid": uid,
                "spectrum_id": spectrum_id,
                "conditions": conditions,
                "frame": normalized,
                "source_rows": group["measurement_source_row"].to_list(),
            }
        )
        audits.append(
            {
                "spectrum_uid": uid,
                "input_rows": group.height,
                "retained_rows": len(retained),
                "excluded_rows": dropped,
                "bds_validation": validation,
            }
        )
    table = pl.concat(frames, how="vertical_relaxed")
    reader_metadata = {
        k: v
        for k, v in read.metadata.items()
        if k in {"sheet_name", "header_row", "encoding", "delimiter", "format"}
    }
    report = {
        "schema_version": "bfl.eis-intake/0.1",
        "provider": bds_distribution()[0],
        "provider_version": bds_distribution()[1],
        "reader_metadata": reader_metadata,
        "group_columns": group_columns,
        "single_spectrum_declared": config.single_spectrum,
        "columns": column_map,
        "phase_unit": phase_unit,
        "polar_conversion": polar,
        "imaginary_sign_flipped": not polar and column_map["imaginary"].strip().startswith("-"),
        "row_index_basis": "zero_based_BDS_parsed_table",
        "input_rows": raw.height,
        "retained_rows": table.height,
        "spectra": audits,
    }
    return spectra, table, report
