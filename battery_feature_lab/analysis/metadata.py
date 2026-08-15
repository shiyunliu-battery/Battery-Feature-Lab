"""Tool-grounded dataset metadata for downstream interpretation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from battery_feature_lab.analysis.adapters import temperature_column
from battery_feature_lab.analysis.schema import AnalysisConfig
from battery_feature_lab.analysis.writer import artifact_reference


def _field(value: Any, *, status: str, source: str, unit: str | None = None) -> dict[str, Any]:
    result = {"value": value, "status": status, "source": source}
    if unit is not None:
        result["unit"] = unit
    return result


def _channel_profile(
    frame: pl.DataFrame,
    *,
    config: AnalysisConfig,
    quantity: str,
    column: str | None,
    unit: str,
    source_column: str | None,
    origin: str,
) -> dict[str, Any]:
    if column is None or column not in frame.columns:
        return {
            "quantity": quantity,
            "analysis_column": column,
            "source_column": source_column,
            "unit": unit,
            "status": "unavailable",
            "origin": origin,
            "finite_sample_fraction": 0.0,
            "observed_min": None,
            "observed_max": None,
            "observed_increment_q10": None,
            "distinct_finite_values": 0,
            "quality_flags": ["channel_unavailable"],
        }
    values = frame[column].cast(pl.Float64, strict=False).to_numpy()
    finite = values[np.isfinite(values)]
    unique = np.unique(finite)
    increments = np.diff(unique)
    scale = max(float(np.nanmax(np.abs(finite))) if len(finite) else 1.0, 1.0)
    increments = increments[
        increments
        > scale * config.policy("metadata_quantization_relative_increment_epsilon")
    ]
    increment_q10 = float(np.quantile(increments, 0.1)) if len(increments) else None
    finite_fraction = float(len(finite) / len(values)) if len(values) else 0.0
    flags: list[str] = []
    if finite_fraction < config.policy("metadata_finite_coverage_warning_min"):
        flags.append("finite_sample_fraction_below_policy_minimum")
    if (
        len(finite) >= int(config.policy("metadata_quantization_min_samples"))
        and len(unique) / len(finite)
        < config.policy("metadata_quantization_unique_fraction_max")
        and increment_q10 is not None
        and increment_q10
        >= (
            config.policy("metadata_voltage_quantization_increment_min_v")
            if quantity == "voltage"
            else config.policy("metadata_nonvoltage_quantization_increment_min")
        )
    ):
        flags.append("coarse_observed_quantization")
    return {
        "quantity": quantity,
        "analysis_column": column,
        "source_column": source_column,
        "unit": unit,
        "status": "available" if len(finite) else "unavailable",
        "origin": origin,
        "finite_sample_fraction": finite_fraction,
        "observed_min": float(np.min(finite)) if len(finite) else None,
        "observed_max": float(np.max(finite)) if len(finite) else None,
        "observed_increment_q10": increment_q10,
        "distinct_finite_values": len(unique),
        "quality_flags": flags,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adjacent_bdf_metadata_artifacts(source: Path) -> list[dict[str, str]]:
    """Reference adjacent BDF metadata files without interpreting their contents."""

    candidates = [
        source.with_name(f"{source.name}.metadata.json"),
        source.with_suffix(".metadata.json"),
        source.with_suffix(".metadata.jsonld"),
        source.with_suffix(".jsonld"),
    ]
    lower_name = source.name.lower()
    for data_suffix in (".bdf.parquet", ".bdf.csv"):
        if lower_name.endswith(data_suffix):
            base_name = source.name[: -len(data_suffix)]
            candidates.extend(
                (
                    source.with_name(f"{base_name}.metadata.json"),
                    source.with_name(f"{base_name}.metadata.jsonld"),
                    source.with_name(f"{base_name}.jsonld"),
                )
            )

    artifacts: list[dict[str, str]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        artifacts.append(
            {
                "filename": candidate.name,
                "sha256": _sha256_file(resolved),
                "role": "bdf_metadata_sidecar",
            }
        )
    return sorted(artifacts, key=lambda item: item["filename"].lower())


def compile_metadata(
    frame: pl.DataFrame,
    report: dict[str, Any],
    *,
    source: Path,
    source_sha256: str,
    run_id: str,
    cell_id: str,
    cycle_id_source: str,
    config: AnalysisConfig,
    channel_details: dict[str, Any],
    input_adapter: str | None = None,
    input_provider: str | None = None,
    input_provider_version: str | None = None,
    input_report_filename: str | None = None,
    handoff_conformance_status: str | None = None,
    adapter_metadata: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile facts supplied by tools or explicit user input; never infer cell facts."""

    include_input_handoff = any(
        value is not None
        for value in (
            input_adapter,
            input_provider,
            input_provider_version,
            input_report_filename,
            handoff_conformance_status,
            adapter_metadata,
            capabilities,
        )
    )
    adapter = (input_adapter or "bds").strip().lower()
    is_native_bdf = adapter in {"bdf", "native_bdf"}
    resolved_provider = input_provider or (
        "batterydf" if is_native_bdf else "battery-data-standard"
    )
    resolved_report_filename = input_report_filename or (
        "bdf_validation_report.json" if is_native_bdf else "bds_conversion_report.json"
    )
    metadata = report.get("metadata", {})
    provenance = {
        item.get("column"): item for item in report.get("provenance", []) if isinstance(item, dict)
    }
    time = frame["test_time_s"].cast(pl.Float64, strict=False).to_numpy()
    finite_time = time[np.isfinite(time)]
    date_column = "Date Time ISO" if "Date Time ISO" in frame.columns else None
    start_datetime = None
    end_datetime = None
    if date_column is not None and frame.height:
        values = frame[date_column].cast(pl.String, strict=False).drop_nulls()
        if len(values):
            start_datetime, end_datetime = values[0], values[-1]

    def provenance_source(canonical: str) -> str | None:
        item = provenance.get(canonical, {})
        return item.get("source") if isinstance(item, dict) else None

    def standardized_origin(*, quantity: str) -> str:
        if not is_native_bdf:
            if quantity == "voltage":
                return str(channel_details.get("voltage_column_source"))
            if quantity == "temperature":
                return str(channel_details.get("temperature_column_source"))
            return "BDS_standardized"
        source_key = f"{quantity}_column_source"
        selection_source = channel_details.get(source_key)
        if selection_source == "explicit":
            return "explicit"
        if selection_source in {"unavailable", "absent"}:
            return "unavailable"
        return "batterydf_standardized"

    def source_column(*, quantity: str, canonical: str) -> str | None:
        if not is_native_bdf:
            return provenance_source(canonical)
        if quantity == "voltage" and channel_details.get("voltage_column_source") == "explicit":
            value = channel_details.get("voltage_column")
            return str(value) if value is not None else None
        if quantity == "temperature" and channel_details.get("temperature_column") is not None:
            value = channel_details.get("temperature_column")
            return str(value) if value is not None else None
        return None

    temp_analysis_column = temperature_column(frame)
    channels = [
        _channel_profile(
            frame,
            config=config,
            quantity="time",
            column="test_time_s",
            unit="s",
            source_column=source_column(quantity="time", canonical="test_time_s"),
            origin=standardized_origin(quantity="time"),
        ),
        _channel_profile(
            frame,
            config=config,
            quantity="voltage",
            column="voltage_v",
            unit="V",
            source_column=(
                str(channel_details.get("voltage_column"))
                if channel_details.get("voltage_column_source") == "explicit"
                else source_column(quantity="voltage", canonical="voltage_v")
            ),
            origin=standardized_origin(quantity="voltage"),
        ),
        _channel_profile(
            frame,
            config=config,
            quantity="current",
            column="current_a",
            unit="A",
            source_column=source_column(quantity="current", canonical="current_a"),
            origin=standardized_origin(quantity="current"),
        ),
        _channel_profile(
            frame,
            config=config,
            quantity="temperature",
            column=temp_analysis_column,
            unit="degC",
            source_column=(
                str(channel_details.get("temperature_column"))
                if channel_details.get("temperature_column") is not None
                else None
            ),
            origin=standardized_origin(quantity="temperature"),
        ),
    ]
    if is_native_bdf:
        cycler = _field(None, status="unknown", source="not_supplied")
        cycler_detection_confidence = _field(
            None, status="unknown", source="not_supplied", unit="1"
        )
        sheet_name = _field(None, status="unknown", source="not_supplied")
        datetime_source = "BDF time-series channel"
        duration_source = "BDF Test Time / s"
        current_sign = "charge-positive"
        raw_current_sign = None
        current_sign_sanity: dict[str, Any] = {}
        adapter_step_cycle = (adapter_metadata or {}).get("step_cycle_semantics", {})
        step_cycle_semantics = (
            dict(adapter_step_cycle) if isinstance(adapter_step_cycle, dict) else {}
        )
        ontology_context = "https://w3id.org/battery-data-alliance/ontology/battery-data-format"
        ontology_use = (
            "formal BDF vocabulary and current-sign reference; no unsupplied entity facts inferred"
        )
        bds_schema_version = None
        bds_adapter_version = None
        bds_support_tier = None
        bds_evidence_tier = None
        source_artifacts = _adjacent_bdf_metadata_artifacts(source)
    else:
        cycler = _field(report.get("cycler"), status="reported", source="BDS")
        cycler_detection_confidence = _field(
            report.get("detection_confidence"), status="reported", source="BDS", unit="1"
        )
        sheet_name = _field(report.get("sheet_name"), status="reported", source="BDS")
        datetime_source = "BDS"
        duration_source = "BDS time channel"
        current_sign = report.get("current_sign")
        raw_current_sign = metadata.get("raw_current_sign")
        current_sign_sanity = metadata.get("current_sign_sanity", {})
        step_cycle_semantics = metadata.get("step_cycle_semantics", {})
        ontology_context = "https://w3id.org/emmo/domain/battery/context"
        ontology_use = "vocabulary_reference_only; no unsupplied entity facts inferred"
        bds_schema_version = report.get("schema_version")
        bds_adapter_version = report.get("adapter_version")
        bds_support_tier = report.get("support_tier")
        bds_evidence_tier = report.get("evidence_tier")
        source_artifacts = []

    source_metadata = (adapter_metadata or {}) if is_native_bdf else metadata
    dataset = {
        "name": source.stem,
        "source_file": artifact_reference(source),
        "source_sha256": source_sha256,
        "source_format": source_metadata.get("source_format", source.suffix.lower().lstrip(".")),
        "row_count": frame.height,
        "bds_schema_version": bds_schema_version,
        "bds_adapter_version": bds_adapter_version,
        "bds_support_tier": bds_support_tier,
        "bds_evidence_tier": bds_evidence_tier,
    }
    if include_input_handoff:
        dataset.update(
            {
                "input_adapter": adapter,
                "provider": resolved_provider,
                "provider_version": input_provider_version,
                "report_file": resolved_report_filename,
                "conformance_status": handoff_conformance_status,
                "adapter_metadata": dict(adapter_metadata or {}),
                "capabilities": dict(capabilities or {}),
                "source_artifacts": source_artifacts,
            }
        )

    tool_sources = [
        {
            "provider": "Battery Feature Lab",
            "role": "compile tool-grounded metadata and observed channel profiles without inferring missing cell facts",
        }
    ]
    if is_native_bdf:
        tool_sources.append(
            {
                "provider": "batterydf",
                "role": (
                    "formal BDF artifact read and validation; charge-positive current convention "
                    "is taken from the BDF specification"
                ),
                "report_file": resolved_report_filename,
            }
        )
    else:
        tool_sources.append(
            {
                "provider": "battery-data-standard",
                "role": "conversion, units, source mapping, sampling and semantic provenance",
                "report_file": resolved_report_filename,
            }
        )
    tool_sources.extend(
        [
            {
                "provider": "PyProBE-Data",
                "role": "operation filters and eligible electrochemical analyses",
                "metadata_status": "bridge object contains no independent source metadata",
            },
            {
                "provider": "BattINFO",
                "role": "semantic vocabulary reference",
                "reference": "https://big-map.github.io/BattINFO/",
            },
            {
                "provider": "Battery Data Toolkit",
                "role": "metadata field-model reference",
                "execution_status": (
                    "not_invoked; input is not a BatteryDataset and the package is not a "
                    "runtime dependency"
                ),
                "reference": (
                    "https://rovi-org.github.io/battery-data-toolkit/user-guide/schemas/"
                    "source-metadata.html"
                ),
            },
        ]
    )

    limitations = [
        "Unknown metadata fields are not inferred from the filename or waveform.",
        "Observed temperature range is not a declared chamber setpoint.",
        "Observed channel increments describe exported values and are not an instrument calibration certificate.",
    ]
    if is_native_bdf and source_artifacts:
        limitations.append(
            "Adjacent BDF metadata sidecars are referenced by filename and digest only; their contents are not interpreted as cell or test facts."
        )

    return {
        "schema_version": "bfl.metadata/0.1",
        "run_id": run_id,
        "cell": {
            "cell_id": _field(
                cell_id,
                status="declared" if config.cell_id else "dataset_identifier",
                source="user" if config.cell_id else "input_file_stem",
            ),
            "chemistry": _field(None, status="unknown", source="not_supplied"),
            "manufacturer": _field(None, status="unknown", source="not_supplied"),
            "design": _field(None, status="unknown", source="not_supplied"),
            "form_factor": _field(None, status="unknown", source="not_supplied"),
            "nominal_capacity": _field(
                config.nominal_capacity_ah,
                status="declared" if config.nominal_capacity_ah is not None else "unknown",
                source="user" if config.nominal_capacity_ah is not None else "not_supplied",
                unit="Ah",
            ),
        },
        "test": {
            "declared_protocol_name": _field(
                config.declared_protocol_name,
                status="declared" if config.declared_protocol_name else "unknown",
                source="user" if config.declared_protocol_name else "not_supplied",
            ),
            "cycler": cycler,
            "cycler_detection_confidence": cycler_detection_confidence,
            "sheet_name": sheet_name,
            "start_datetime": _field(
                start_datetime,
                status="reported" if start_datetime else "unknown",
                source=(datetime_source if start_datetime or not is_native_bdf else "not_supplied"),
            ),
            "end_datetime": _field(
                end_datetime,
                status="reported" if end_datetime else "unknown",
                source=datetime_source if end_datetime or not is_native_bdf else "not_supplied",
            ),
            "observed_duration": _field(
                float(finite_time[-1] - finite_time[0]) if len(finite_time) >= 2 else None,
                status="observed" if len(finite_time) >= 2 else "unknown",
                source=duration_source,
                unit="s",
            ),
            "set_temperature": _field(None, status="unknown", source="not_supplied", unit="degC"),
        },
        "dataset": dataset,
        "channels": channels,
        "semantics": {
            "current_sign": current_sign,
            "raw_current_sign": raw_current_sign,
            "current_sign_sanity": current_sign_sanity,
            "cycle_id_source": cycle_id_source,
            "step_cycle_semantics": step_cycle_semantics,
            "ontology_context": ontology_context,
            "ontology_use": ontology_use,
        },
        "tool_sources": tool_sources,
        "limitations": limitations,
    }
