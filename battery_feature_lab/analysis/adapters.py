"""Standardized input handoffs and the narrow PyProBE object bridge."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bds
import numpy as np
import polars as pl
import pyprobe
from pyprobe.filters import Experiment

from battery_feature_lab.analysis.numerics import integrate_previous, reported_delta

_BDF_TO_CANONICAL = {
    "Test Time / s": "test_time_s",
    "Voltage / V": "voltage_v",
    "Current / A": "current_a",
    "Cycle Count / 1": "cycle_index",
    "Step Count / 1": "step_index",
    "Step Index / 1": "record_index",
    "Step Time / s": "step_time_s",
    "Power / W": "power_w",
    "Ambient Temperature / degC": "ambient_temperature_deg_c",
    "Surface Temperature T1 / degC": "temperature_t1_deg_c",
    "Charging Capacity / Ah": "charge_capacity_ah",
    "Discharging Capacity / Ah": "discharge_capacity_ah",
    "Charging Energy / Wh": "charge_energy_wh",
    "Discharging Energy / Wh": "discharge_energy_wh",
    "Internal Resistance / ohm": "internal_resistance_ohm",
}

_BDF_MACHINE_TO_CANONICAL = {
    "test_time_second": "test_time_s",
    "voltage_volt": "voltage_v",
    "current_ampere": "current_a",
    "cycle_count": "cycle_index",
    "step_count": "step_index",
    "step_index": "record_index",
    "ambient_temperature_celsius": "ambient_temperature_deg_c",
    "charging_capacity_ah": "charge_capacity_ah",
    "discharging_capacity_ah": "discharge_capacity_ah",
    "charging_energy_wh": "charge_energy_wh",
    "discharging_energy_wh": "discharge_energy_wh",
    "power_watt": "power_w",
    "internal_resistance_ohm": "internal_resistance_ohm",
}


@dataclass(frozen=True)
class InputHandoff:
    """One tool-validated handoff into the BFL analysis core."""

    frame: pl.DataFrame
    adapter: str
    provider: str
    provider_version: str
    report: Any
    report_payload: dict[str, Any]
    report_path: Path
    conformance_status: str
    adapter_metadata: dict[str, Any]


def prepare_input_handoff(
    input_path: Path,
    normalized_path: Path,
    output_dir: Path,
    *,
    input_adapter: str,
) -> InputHandoff:
    """Use BDS for raw conversion or formal BDF for an existing BDF artifact."""

    selected = detect_input_adapter(input_path) if input_adapter == "auto" else input_adapter
    if selected == "bds":
        report_path = output_dir / "bds_conversion_report.json"
        frame, report = convert_with_bds(input_path, normalized_path, report_path)
        payload = report_dict(report)
        conformance = _validate_bds_partial_handoff(frame, payload)
        return InputHandoff(
            frame=frame,
            adapter="bds",
            provider="battery-data-standard",
            provider_version=importlib.metadata.version("battery-data-standard"),
            report=report,
            report_payload=payload,
            report_path=report_path,
            conformance_status=conformance,
            adapter_metadata=dict(payload.get("metadata", {})),
        )
    if selected == "bdf":
        report_path = output_dir / "bdf_validation_report.json"
        frame, report_payload, adapter_metadata, provider_version = convert_with_bdf(
            input_path,
            normalized_path,
            report_path,
        )
        return InputHandoff(
            frame=frame,
            adapter="bdf",
            provider="batterydf",
            provider_version=provider_version,
            report=report_payload,
            report_payload=report_payload,
            report_path=report_path,
            conformance_status="batterydf_validator_passed",
            adapter_metadata=adapter_metadata,
        )
    raise ValueError("input_adapter must be one of: auto, bds, bdf")


def detect_input_adapter(input_path: Path) -> str:
    """Choose formal BDF only for an artifact with explicit BDF identity."""

    # A BFL artifact keeps its original handoff identity in the sibling
    # validation file. This prevents a BDS legacy-shape export from being
    # reclassified as formal BDF solely because of its filename.
    if input_path.name == "normalized_data.bdf.parquet":
        validation_path = input_path.parent / "analysis_validation.json"
        if validation_path.is_file():
            try:
                payload = json.loads(validation_path.read_text(encoding="utf-8"))
                previous = payload.get("recomputation", {}).get("input_adapter")
                if previous in {"bds", "bdf"}:
                    return str(previous)
            except (OSError, ValueError, TypeError):
                pass
    if ".bdf." in input_path.name.lower():
        return "bdf"
    try:
        suffix = input_path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            columns = set(pl.read_parquet_schema(input_path))
        elif suffix == ".csv":
            columns = set(pl.scan_csv(input_path, n_rows=0).collect_schema().names())
        else:
            return "bds"
    except Exception:  # noqa: BLE001 - routing falls back without mutating input
        return "bds"
    preferred = {"Test Time / s", "Voltage / V", "Current / A"}
    machine = {"test_time_second", "voltage_volt", "current_ampere"}
    return "bdf" if preferred <= columns or machine <= columns else "bds"


def convert_with_bdf(
    input_path: Path,
    normalized_path: Path,
    report_path: Path,
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, Any], str]:
    """Validate and load an existing formal BDF artifact with ``batterydf``."""

    try:
        bdf = importlib.import_module("bdf")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "formal BDF input requires the optional dependency; install battery-feature-lab[bdf]"
        ) from exc

    loaded = bdf.read(input_path, normalize=True, validate=True)
    adapter_metadata: dict[str, Any] = {}
    if isinstance(loaded, tuple) and len(loaded) == 2:
        table, metadata = loaded
        adapter_metadata["provider_api_variant"] = "tuple_dataframe_and_metadata"
        if isinstance(metadata, dict):
            adapter_metadata["provider_read_metadata"] = dict(metadata)
    else:
        table = loaded
        adapter_metadata["provider_api_variant"] = "pypi_0.1_dataframe"
    ontology_overrides = {
        name: value for name in ("BDF_ONTOLOGY_PATH", "BDF_ONTOLOGY") if (value := os.getenv(name))
    }
    if ontology_overrides:
        adapter_metadata["ontology_environment_overrides"] = ontology_overrides

    validation = bdf.validate(table, report=False, raise_on_error=False)
    if not isinstance(validation, dict) or not bool(validation.get("ok")):
        detail = validation.get("detail") if isinstance(validation, dict) else None
        raise ValueError(f"formal BDF validation failed: {detail or validation}")

    # The released 0.1 API accepts pandas while the current API accepts pandas
    # or Polars. Keep the provider-owned object to avoid duplicating its writer.
    save = getattr(bdf, "save", None)
    if save is None:
        save = importlib.import_module("bdf.io").save
    save_parameters = inspect.signature(save).parameters
    save_kwargs: dict[str, Any] = {}
    if "labels" in save_parameters:
        save_kwargs["labels"] = "preferred"
    elif "human" in save_parameters:
        save_kwargs["human"] = True
    save(table, normalized_path, **save_kwargs)

    exported = pl.read_parquet(normalized_path)
    labels = _BDF_TO_CANONICAL | _BDF_MACHINE_TO_CANONICAL
    rename = {name: labels[name] for name in exported.columns if name in labels}
    canonical = exported.rename(rename).with_row_index("_source_row")
    serializable_validation = json.loads(
        json.dumps(validation, ensure_ascii=False, allow_nan=False)
    )
    report_path.write_text(
        json.dumps(serializable_validation, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    try:
        provider_version = importlib.metadata.version("batterydf")
    except importlib.metadata.PackageNotFoundError:
        provider_version = str(getattr(bdf, "__version__", "unknown"))
    return canonical, serializable_validation, adapter_metadata, provider_version


def _validate_bds_partial_handoff(frame: pl.DataFrame, payload: dict[str, Any]) -> str:
    """Permit only measurement-channel omissions in a BDS partial handoff."""

    columns = set(frame.columns)
    missing_core = sorted({"test_time_s", "current_a"} - columns)
    if missing_core:
        raise ValueError(f"BDS handoff is missing BFL core channels: {missing_core}")
    for column in ("test_time_s", "current_a"):
        values = frame[column].cast(pl.Float64, strict=False)
        if not bool((values.is_not_null() & values.is_finite()).any()):
            raise ValueError(f"BDS handoff has no finite {column} values")

    validation = payload.get("validation", {})
    errors = [
        issue
        for issue in validation.get("issues", [])
        if isinstance(issue, dict) and issue.get("level") == "error"
    ]
    allowed = [
        issue
        for issue in errors
        if issue.get("code") == "missing-required-column" and issue.get("column") == "voltage_v"
    ]
    unexpected = [issue for issue in errors if issue not in allowed]
    if unexpected:
        raise ValueError(f"BDS handoff contains non-degradable validation errors: {unexpected}")
    return (
        "bds_valid_legacy_bdf_style"
        if bool(validation.get("valid"))
        else "bds_partial_legacy_bdf_style"
    )


def convert_with_bds(
    input_path: Path,
    normalized_path: Path,
    report_path: Path,
) -> tuple[pl.DataFrame, Any]:
    """Create the user-facing BDF Parquet and unwrapped BDS JSON report."""

    report = bds.convert(
        input_path,
        normalized_path,
        format="parquet",
        cycler="auto",
        strict=False,
        keep_raw=True,
        current_sign="charge-positive",
        repair_policy="warn",
        time_sampling_policy="warn",
        current_sign_check="none",
        target="bdf",
        write_sidecars=False,
    )
    report.write_json(report_path)
    exported = pl.read_parquet(normalized_path)
    rename = {
        name: _BDF_TO_CANONICAL[name] for name in exported.columns if name in _BDF_TO_CANONICAL
    }
    canonical = exported.rename(rename).with_row_index("_source_row")
    return canonical, report


def report_dict(report: Any) -> dict[str, Any]:
    """Return the exact public serialization of a BDS report."""

    return dict(report.to_dict())


def prepare_analysis_frame(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Exclude rows that cannot be located in time without changing the BDF export."""

    if "test_time_s" not in frame.columns:
        raise ValueError("BDS output is missing the required test_time_s column")
    time = frame["test_time_s"].cast(pl.Float64, strict=False)
    valid_time = time.is_not_null() & time.is_finite()
    excluded = frame.filter(~valid_time)
    usable = frame.filter(valid_time).sort("test_time_s")
    if usable.is_empty():
        raise ValueError("BDS output has no rows with finite test time")
    excluded_rows = (
        excluded["_source_row"].cast(pl.Int64, strict=False).drop_nulls().to_list()
        if "_source_row" in excluded.columns
        else []
    )
    return usable, {
        "normalized_row_count": frame.height,
        "analysis_row_count": usable.height,
        "excluded_rows_without_finite_time": excluded.height,
        "excluded_source_rows_without_finite_time": excluded_rows,
    }


def cycle_source(frame: pl.DataFrame, report: Any) -> str:
    """Classify the cycle identifier source using the BDS conversion report."""

    if "cycle_index" not in frame.columns:
        return "absent"
    metadata = getattr(report, "metadata", {}) or {}
    semantic = metadata.get("semantic_sources", {})
    step_cycle = metadata.get("step_cycle_semantics", {})
    cycle_semantic = semantic.get("cycle_index") or step_cycle.get("semantic_sources", {}).get(
        "cycle_index", {}
    )
    origin = str(cycle_semantic.get("origin", "")).lower()
    if origin in {"source", "joined", "inferred", "absent"}:
        return origin
    provenance = getattr(report, "provenance", []) or []
    for item in provenance:
        column = getattr(item, "column", None)
        source = getattr(item, "source", None)
        if column == "cycle_index" and source:
            return "source"
    return "unknown"


def handoff_cycle_source(handoff: InputHandoff) -> str:
    """Return a conservative cycle identifier source for either input adapter."""

    if handoff.adapter == "bds":
        return cycle_source(handoff.frame, handoff.report)
    return "source" if "cycle_index" in handoff.frame.columns else "absent"


def channel_capabilities(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Inventory independent measurement capabilities after preprocessing."""

    mapping = {
        "time": "test_time_s",
        "current": "current_a",
        "voltage": "voltage_v",
        "temperature": temperature_column(frame),
        "cycle": "cycle_index",
        "step": "step_index",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, column in mapping.items():
        exists = column is not None and column in frame.columns
        if name in {"cycle", "step"}:
            finite_fraction = None
            usable = bool(exists and frame[column].drop_nulls().len())
        elif exists:
            values = frame[column].cast(pl.Float64, strict=False)
            finite_fraction = float((values.is_not_null() & values.is_finite()).mean())
            usable = finite_fraction > 0.0
        else:
            finite_fraction = 0.0
            usable = False
        result[name] = {
            "column": column,
            "present": bool(exists),
            "usable": bool(usable),
            "finite_sample_fraction": finite_fraction,
        }
    return result


def temperature_column(frame: pl.DataFrame) -> str | None:
    """Select the first standardized temperature channel without guessing raw columns."""

    for name in (
        "analysis_temperature_deg_c",
        "temperature_t1_deg_c",
        "ambient_temperature_deg_c",
        "surface_temperature_celsius",
    ):
        if name in frame.columns:
            return name
    return None


def apply_analysis_channel_overrides(
    frame: pl.DataFrame,
    *,
    voltage_column: str | None,
    temperature_column_name: str | None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply exact, user-declared analysis channels without modifying the BDF file.

    Raw columns are never guessed.  The returned aliases exist only in the
    in-memory analysis frame and the selection is carried into validation.
    """

    selected_voltage = "voltage_v" if "voltage_v" in frame.columns else None
    selected_temperature = temperature_column(frame)
    output = frame
    if voltage_column is not None:
        if voltage_column not in frame.columns:
            raise ValueError(
                f"configured voltage_column is absent from BDS output: {voltage_column}"
            )
        output = output.with_columns(
            pl.col(voltage_column).cast(pl.Float64, strict=False).alias("voltage_v")
        )
        selected_voltage = voltage_column
    if temperature_column_name is not None:
        if temperature_column_name not in frame.columns:
            raise ValueError(
                "configured temperature_column is absent from BDS output: "
                f"{temperature_column_name}"
            )
        output = output.with_columns(
            pl.col(temperature_column_name)
            .cast(pl.Float64, strict=False)
            .alias("analysis_temperature_deg_c")
        )
        selected_temperature = temperature_column_name

    def coverage(column: str | None) -> float | None:
        if column is None:
            return None
        alias = (
            "voltage_v"
            if column == selected_voltage
            else "analysis_temperature_deg_c"
            if temperature_column_name is not None and column == selected_temperature
            else column
        )
        if alias not in output.columns:
            return None
        values = output[alias].cast(pl.Float64, strict=False)
        return float((values.is_not_null() & values.is_finite()).mean())

    return output, {
        "voltage_column": selected_voltage,
        "voltage_column_source": (
            "explicit" if voltage_column else "standardized" if selected_voltage else "unavailable"
        ),
        "voltage_finite_sample_fraction": coverage(selected_voltage),
        "temperature_column": selected_temperature,
        "temperature_column_source": (
            "explicit"
            if temperature_column_name
            else "BDS_standardized"
            if selected_temperature
            else "unavailable"
        ),
        "temperature_finite_sample_fraction": coverage(selected_temperature),
        "bdf_file_modified": False,
    }


def prepare_pyprobe_frame(
    frame: pl.DataFrame,
    *,
    rest_current_threshold_a: float,
    sampling_interval_outlier_factor: float,
    reported_capacity_finite_coverage_min: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Construct the exact columns required by PyProBE without changing BDF data."""

    if frame.is_empty():
        raise ValueError("cannot construct a PyProBE object from an empty frame")
    required = {"test_time_s", "voltage_v", "current_a"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"BDS output is missing PyProBE-required columns: {missing}")

    ordered = frame.sort("test_time_s")
    time = ordered["test_time_s"].cast(pl.Float64).to_numpy()
    current = ordered["current_a"].cast(pl.Float64).to_numpy()
    zeroed_current = np.where(
        np.isfinite(current) & (np.abs(current) <= rest_current_threshold_a), 0.0, current
    )

    capacity, capacity_method, capacity_flags = _net_capacity(
        ordered,
        time,
        current,
        sampling_interval_outlier_factor,
        reported_capacity_finite_coverage_min,
    )
    if "step_index" in ordered.columns:
        raw_step = (
            ordered["step_index"]
            .cast(pl.Int64, strict=False)
            .fill_null(strategy="forward")
            .fill_null(0)
        )
        step_values = raw_step.to_numpy()
    else:
        phase = np.where(zeroed_current == 0, 0, np.where(zeroed_current > 0, 1, -1))
        step_values = np.cumsum(np.r_[0, phase[1:] != phase[:-1]]).astype(np.int64)
    event_values = np.cumsum(np.r_[0, step_values[1:] != step_values[:-1]]).astype(np.int64)
    if "cycle_index" in ordered.columns:
        cycle_values = (
            ordered["cycle_index"]
            .cast(pl.Int64, strict=False)
            .fill_null(strategy="forward")
            .fill_null(0)
            .to_numpy()
        )
    else:
        cycle_values = np.zeros(len(ordered), dtype=np.int64)

    source_record = (
        ordered.select(
            pl.coalesce(
                pl.col("record_index").cast(pl.Int64, strict=False),
                pl.col("_source_row").cast(pl.Int64),
            ).alias("_source_record")
        )["_source_record"].to_numpy()
        if "record_index" in ordered.columns
        else ordered["_source_row"].to_numpy()
    )
    output = pl.DataFrame(
        {
            "Time [s]": time,
            "Step": step_values,
            "Cycle": cycle_values,
            "Event": event_values,
            "Current [A]": zeroed_current,
            "Voltage [V]": ordered["voltage_v"].cast(pl.Float64).to_numpy(),
            "Capacity [Ah]": capacity,
            "_source_row": ordered["_source_row"].to_numpy(),
            "_source_record": source_record,
        }
    )
    temp = temperature_column(ordered)
    if temp is not None:
        output = output.with_columns(
            pl.Series("Temperature [C]", ordered[temp].cast(pl.Float64, strict=False).to_numpy())
        )
    return output, {
        "capacity_method": capacity_method,
        "capacity_flags": capacity_flags,
        "reported_capacity_finite_coverage_min": reported_capacity_finite_coverage_min,
    }


def analysis_capacity(
    frame: pl.DataFrame,
    *,
    sampling_interval_outlier_factor: float,
    reported_capacity_finite_coverage_min: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose a declared capacity branch for one ordered analysis scope."""

    ordered = frame.sort("test_time_s")
    time = ordered["test_time_s"].cast(pl.Float64).to_numpy()
    current = ordered["current_a"].cast(pl.Float64).to_numpy()
    values, method, flags = _net_capacity(
        ordered,
        time,
        current,
        sampling_interval_outlier_factor,
        reported_capacity_finite_coverage_min,
    )
    return values, {
        "capacity_method": method,
        "capacity_flags": flags,
        "reported_capacity_finite_coverage_min": reported_capacity_finite_coverage_min,
    }


def pyprobe_experiment(frame: pl.DataFrame) -> Experiment:
    """Create a one-scope PyProBE Experiment without cycle inference."""

    visible = frame.select(
        [
            "Time [s]",
            "Step",
            "Cycle",
            "Event",
            "Current [A]",
            "Voltage [V]",
            "Capacity [Ah]",
            *(["Temperature [C]"] if "Temperature [C]" in frame.columns else []),
            *(["_source_row"] if "_source_row" in frame.columns else []),
            *(["_source_record"] if "_source_record" in frame.columns else []),
        ]
    )
    steps = visible["Step"].cast(pl.Int64).to_numpy()
    cycle_info = [(int(np.min(steps)), int(np.max(steps)), 1)] if len(steps) else []
    return Experiment(lf=visible.lazy(), info={}, cycle_info=cycle_info)


def pyprobe_result(frame: pl.DataFrame, columns: list[str]) -> pyprobe.Result:
    """Create a generic PyProBE Result for public analysis functions."""

    definitions = {name.split(" [", 1)[0]: name for name in columns}
    return pyprobe.Result(lf=frame.select(columns).lazy(), info={}, column_definitions=definitions)


def _net_capacity(
    frame: pl.DataFrame,
    time: np.ndarray,
    current: np.ndarray,
    sampling_interval_outlier_factor: float,
    reported_capacity_finite_coverage_min: float,
) -> tuple[np.ndarray, str, list[str]]:
    flags: list[str] = []
    if {"charge_capacity_ah", "discharge_capacity_ah"} <= set(frame.columns):
        charge = frame["charge_capacity_ah"].cast(pl.Float64, strict=False).to_numpy()
        discharge = frame["discharge_capacity_ah"].cast(pl.Float64, strict=False).to_numpy()
        _, charge_flags = reported_delta(charge)
        _, discharge_flags = reported_delta(discharge)
        if not charge_flags and not discharge_flags:
            net = charge - discharge
            if (
                float(np.mean(np.isfinite(net)))
                >= reported_capacity_finite_coverage_min
            ):
                return net, "reported_charge_minus_discharge", []
            flags.append("reported_capacity_columns_have_missing_values")
        flags.extend(charge_flags + discharge_flags)

    increments = np.zeros(len(time), dtype=float)
    if len(time) >= 2:
        dt = np.diff(time)
        valid = np.isfinite(dt) & (dt > 0) & np.isfinite(current[:-1])
        increments[1:] = np.where(valid, current[:-1] * dt / 3600.0, 0.0)
    integrated, interval = integrate_previous(
        time,
        current,
        scale=1.0 / 3600.0,
        sampling_interval_outlier_factor=sampling_interval_outlier_factor,
    )
    if integrated is None:
        flags.append("capacity_not_computable")
    if interval.sampling_interval_outlier_count:
        flags.append("capacity_includes_sampling_interval_outlier")
    return np.cumsum(increments), "zoh_previous_v1", sorted(set(flags))
