"""Orchestration for the single Battery Feature Lab analysis contract."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import bds
import numpy
import polars
import pyprobe
import scipy

from battery_feature_lab.analysis.adapters import (
    apply_analysis_channel_overrides,
    channel_capabilities,
    detect_input_adapter,
    handoff_cycle_source,
    prepare_analysis_frame,
    prepare_input_handoff,
    prepare_pyprobe_frame,
)
from battery_feature_lab.analysis.characterization import (
    analyze_current_step_summary,
    analyze_current_steps,
    analyze_directional_energy,
    analyze_relaxation_summary,
)
from battery_feature_lab.analysis.evolution import analyze_capacity_evolution
from battery_feature_lab.analysis.metadata import compile_metadata
from battery_feature_lab.analysis.operation import analyze_operation
from battery_feature_lab.analysis.profiles import analyze_capacity_aligned_profile
from battery_feature_lab.analysis.response import (
    analyze_cycles,
    analyze_ica_dva,
    analyze_pulses,
    analyze_rest_and_thermal,
    record_counts,
)
from battery_feature_lab.analysis.schema import (
    ANALYSIS_POLICY_VERSION,
    METADATA_SCHEMA,
    RESULTS_SCHEMA,
    SCHEMA_VERSION,
    SUMMARY_SCHEMA,
    VALIDATION_SCHEMA,
    VALIDATION_SCHEMA_VERSION,
    AnalysisConfig,
    AnalysisResult,
    make_record,
    metric,
    seconds_label,
)
from battery_feature_lab.analysis.summary import compile_summary
from battery_feature_lab.analysis.writer import (
    canonical_sha256,
    sha256_file,
    validate_payload,
    write_json,
)


def compile_analysis(input_path: Path, config: AnalysisConfig) -> AnalysisResult:
    """Run BDS ingest, tool-backed analysis, validation, and structured export."""

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = config.output_dir.expanduser().resolve()
    selected_adapter = (
        detect_input_adapter(source) if config.input_adapter == "auto" else config.input_adapter
    )
    report_filename = (
        "bdf_validation_report.json" if selected_adapter == "bdf" else "bds_conversion_report.json"
    )
    _prepare_output_directory(output_dir, report_filename)
    normalized_path = output_dir / "normalized_data.bdf.parquet"
    metadata_path = output_dir / "analysis_metadata.json"
    results_path = output_dir / "analysis_results.json"
    evidence_path = output_dir / "analysis_evidence.json"
    validation_path = output_dir / "analysis_validation.json"

    provider_calls: list[dict[str, Any]] = []
    handoff = prepare_input_handoff(
        source,
        normalized_path,
        output_dir,
        input_adapter=selected_adapter,
    )
    report_path = handoff.report_path
    frame = handoff.frame
    frame, analysis_frame_details = prepare_analysis_frame(frame)
    frame, channel_details = apply_analysis_channel_overrides(
        frame,
        voltage_column=config.voltage_column,
        temperature_column_name=config.temperature_column,
    )
    provider_calls.append(
        {
            "provider": handoff.provider,
            "method": "convert" if handoff.adapter == "bds" else "read_validate",
            "status": "ok",
            "input_path": str(source),
            "output_file": normalized_path.name,
            "parameters": {
                "input_adapter": handoff.adapter,
                "target": "legacy_bdf_style" if handoff.adapter == "bds" else "formal_bdf",
                "format": "parquet",
                **(
                    {
                        "repair_policy": "warn",
                        "time_sampling_policy": "warn",
                        "current_sign": "charge-positive",
                    }
                    if handoff.adapter == "bds"
                    else {"current_sign": "charge-positive_from_bdf_spec"}
                ),
            },
            "report_file": report_path.name,
            "handoff_conformance_status": handoff.conformance_status,
        }
    )
    provider_calls.append(
        {
            "provider": f"{handoff.provider}+BFL",
            "method": "select_analysis_channels",
            "status": "ok",
            **channel_details,
        }
    )
    direct_report = handoff.report_payload
    with report_path.open(encoding="utf-8") as stream:
        on_disk_report = json.load(stream)
    if on_disk_report != direct_report:
        raise RuntimeError("input report file differs from the provider-native report payload")

    capabilities = channel_capabilities(frame)
    pyprobe_frame: polars.DataFrame | None
    if capabilities["voltage"]["usable"]:
        pyprobe_frame, adapter_details = prepare_pyprobe_frame(
            frame,
            rest_current_threshold_a=config.rest_current_threshold_a,
            sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
            reported_capacity_finite_coverage_min=config.policy(
                "reported_capacity_finite_coverage_min"
            ),
        )
        provider_calls.append(
            {
                "provider": "BFL",
                "method": "standardized_to_pyprobe_bridge",
                "status": "ok",
                "input_adapter": handoff.adapter,
                "input_rows": analysis_frame_details["normalized_row_count"],
                "output_rows": pyprobe_frame.height,
                **analysis_frame_details,
                **adapter_details,
            }
        )
    else:
        pyprobe_frame = None
        adapter_details = {"capacity_method": None, "capacity_flags": []}
        provider_calls.append(
            {
                "provider": "BFL",
                "method": "standardized_to_pyprobe_bridge",
                "status": "not_invoked",
                "reason": "missing_required_channel:voltage_v",
                **analysis_frame_details,
            }
        )

    cell_id = config.cell_id or source.stem
    cycle_id_source = handoff_cycle_source(handoff)
    if pyprobe_frame is None:
        cycle_records = [
            _missing_channel_record(
                "response.cycle_summary",
                cell_id,
                reason="missing_required_channel:voltage_v",
                source_interval=_frame_interval(frame),
                provider="PyProBE",
                method="cycling.summary",
            )
        ]
        cycle_summaries: list[dict[str, Any]] = []
        provider_calls.append(
            {
                "provider": "PyProBE",
                "method": "cycling.summary",
                "status": "not_invoked",
                "reason": "missing_required_channel:voltage_v",
                "source_interval": _frame_interval(frame),
            }
        )
    else:
        cycle_records, cycle_summaries = analyze_cycles(
            frame,
            pyprobe_frame,
            config=config,
            cell_id=cell_id,
            cycle_id_source=cycle_id_source,
            provider_calls=provider_calls,
        )
    representative_cycle, representative_reason = _select_representative_cycle(
        config, cycle_summaries
    )

    # The default operating summary describes the complete observation window.
    # A representative cycle is reserved for analyses whose reference frame is
    # explicitly cycle-specific.
    short_window_cycle: int | None = None
    operation_records, short_window_phases = analyze_operation(
        frame,
        pyprobe_frame,
        config=config,
        cell_id=cell_id,
        cycle_id=short_window_cycle,
        provider_calls=provider_calls,
    )

    # Cycle-specific pulse/ICA/DVA analysis needs phases aligned to the selected
    # cycle. Reuse the whole-window phases only when no representative cycle is
    # available; otherwise derive a separate phase view without emitting a
    # second set of operation records.
    if representative_cycle is None:
        cycle_phases = short_window_phases
    else:
        _, cycle_phases = analyze_operation(
            frame,
            pyprobe_frame,
            config=config,
            cell_id=cell_id,
            cycle_id=representative_cycle,
            provider_calls=provider_calls,
        )

    records = [*operation_records, *cycle_records]
    if pyprobe_frame is not None:
        rest_records = analyze_rest_and_thermal(
            frame,
            pyprobe_frame,
            short_window_phases,
            config=config,
            cell_id=cell_id,
            cycle_id=short_window_cycle,
            provider_calls=provider_calls,
        )
        records.extend(rest_records)
        records.append(
            analyze_relaxation_summary(
                rest_records,
                config=config,
                cell_id=cell_id,
                cycle_id=short_window_cycle,
            )
        )
        records.append(
            analyze_directional_energy(
                frame,
                config=config,
                cell_id=cell_id,
                cycle_id=short_window_cycle,
            )
        )
        records.append(
            analyze_capacity_aligned_profile(
                frame,
                short_window_phases,
                config=config,
                cell_id=cell_id,
                cycle_id=short_window_cycle,
                cycle_id_source=cycle_id_source,
            )
        )
        current_step_times = tuple(sorted({2.0, *config.pulse_resistance_times_s}))
        current_step_records = analyze_current_steps(
            frame,
            short_window_phases,
            response_times_s=current_step_times,
            config=config,
            cell_id=cell_id,
            cycle_id=short_window_cycle,
        )
        records.extend(current_step_records)
        records.append(
            analyze_current_step_summary(
                current_step_records,
                response_times_s=current_step_times,
                config=config,
                cell_id=cell_id,
                cycle_id=short_window_cycle,
            )
        )
        records.extend(
            analyze_pulses(
                frame,
                pyprobe_frame,
                cycle_phases,
                cycle_summaries,
                config=config,
                cell_id=cell_id,
                cycle_id=representative_cycle,
                provider_calls=provider_calls,
            )
        )
        records.extend(
            analyze_ica_dva(
                frame,
                pyprobe_frame,
                cycle_phases,
                cycle_summaries,
                config=config,
                cell_id=cell_id,
                cycle_id=representative_cycle,
                provider_calls=provider_calls,
            )
        )
    else:
        reason = "missing_required_channel:voltage_v"
        full_interval = _frame_interval(frame)
        for record_type, provider, method in (
            ("response.rest_and_thermal", "PyProBE+BFL", "rest_filter_with_descriptors"),
            ("response.relaxation_signature", "BFL", "fixed_time_relaxation_summary_v1"),
            (
                "response.directional_energy_summary",
                "BFL",
                "directional_previous_zoh_energy_voltage_v1",
            ),
            (
                "response.capacity_aligned_profile",
                "BFL",
                "capacity_aligned_previous_zoh_linear_v1",
            ),
            ("response.current_step", "BFL", "apparent_current_step_response_v1"),
            ("response.current_step_summary", "BFL", "current_step_population_summary_v1"),
            ("response.pulse_resistance", "PyProBE", "pulsing.get_resistances"),
            ("response.ica_curve", "PyProBE", "differentiation.differentiate_lean"),
            ("response.dva_curve", "PyProBE", "differentiation.differentiate_lean"),
        ):
            records.append(
                _missing_channel_record(
                    record_type,
                    cell_id,
                    reason=reason,
                    source_interval=full_interval,
                    provider=provider,
                    method=method,
                )
            )
        for method in ("rest", "pulsing.get_resistances", "differentiation.differentiate_lean"):
            provider_calls.append(
                {
                    "provider": "PyProBE",
                    "method": method,
                    "status": "not_invoked",
                    "reason": reason,
                    "source_interval": full_interval,
                }
            )
    records.append(
        analyze_capacity_evolution(
            cycle_summaries,
            config=config,
            cell_id=cell_id,
            provider_calls=provider_calls,
        )
    )
    record_ids = [record["record_id"] for record in records]
    seen_record_ids: set[str] = set()
    duplicate_record_ids: set[str] = set()
    for record_id in record_ids:
        if record_id in seen_record_ids:
            duplicate_record_ids.add(record_id)
        seen_record_ids.add(record_id)
    if duplicate_record_ids:
        raise ValueError(f"duplicate analysis record IDs: {sorted(duplicate_record_ids)}")

    run_id = _run_id(source, config)
    configuration = _configuration_dict(config)
    configuration["selected_representative_cycle"] = representative_cycle
    configuration["representative_cycle_selection"] = representative_reason
    source_sha256 = sha256_file(source)
    metadata_payload = compile_metadata(
        frame,
        direct_report,
        source=source,
        source_sha256=source_sha256,
        run_id=run_id,
        cell_id=cell_id,
        cycle_id_source=cycle_id_source,
        config=config,
        channel_details=channel_details,
        input_adapter=handoff.adapter,
        input_provider=handoff.provider,
        input_provider_version=handoff.provider_version,
        input_report_filename=report_path.name,
        handoff_conformance_status=handoff.conformance_status,
        adapter_metadata=handoff.adapter_metadata,
        capabilities=capabilities,
    )
    metadata_errors = validate_payload(metadata_payload, METADATA_SCHEMA)
    if metadata_errors:
        raise ValueError("analysis_metadata.json schema errors: " + "; ".join(metadata_errors))
    write_json(metadata_path, metadata_payload)

    channel_profiles = {item["quantity"]: item for item in metadata_payload["channels"]}
    voltage_is_coarse = "coarse_observed_quantization" in channel_profiles.get("voltage", {}).get(
        "quality_flags", []
    )
    resolution_sensitive_types = {
        "response.current_step",
        "response.current_step_summary",
        "response.rest_and_thermal",
        "response.relaxation_signature",
        "response.capacity_aligned_profile",
        "response.ica_curve",
        "response.dva_curve",
    }
    for record in records:
        record["attributes"]["metadata_context"] = {
            "file": metadata_path.name,
            "run_id": run_id,
        }
        if voltage_is_coarse and record["record_type"] in resolution_sensitive_types:
            flags = record["quality"]["flags"]
            if "coarse_voltage_quantization" not in flags:
                flags.append("coarse_voltage_quantization")
                flags.sort()
            record["quality"]["status"] = "warning"

    evidence_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "cell_id": cell_id,
        "source": {
            "input_path": str(source),
            "input_sha256": source_sha256,
            "normalized_data_file": normalized_path.name,
            "input_adapter": handoff.adapter,
            "input_provider": handoff.provider,
            "input_report_file": report_path.name,
            "handoff_conformance_status": handoff.conformance_status,
            "metadata_file": metadata_path.name,
            "cycle_id_source": cycle_id_source,
        },
        "configuration": configuration,
        "records": records,
    }
    result_errors = validate_payload(evidence_payload, RESULTS_SCHEMA)
    if result_errors:
        raise ValueError("analysis_evidence.json schema errors: " + "; ".join(result_errors))
    write_json(evidence_path, evidence_payload)

    results_payload = compile_summary(records, run_id=run_id, cell_id=cell_id)
    summary_errors = validate_payload(results_payload, SUMMARY_SCHEMA)
    if summary_errors:
        raise ValueError("analysis_results.json schema errors: " + "; ".join(summary_errors))
    write_json(results_path, results_payload)

    analysis_row_warnings = (
        {"rows_without_finite_time_excluded_from_analysis"}
        if analysis_frame_details["excluded_rows_without_finite_time"]
        else set()
    )
    warnings = sorted(
        {flag for record in records for flag in record["quality"]["flags"]}
        | set(direct_report.get("warnings", []))
        | analysis_row_warnings
    )
    provider_errors = [call for call in provider_calls if call.get("status") == "error"]
    artifacts = {
        normalized_path.name: _artifact(normalized_path),
        report_path.name: _artifact(report_path),
        metadata_path.name: _artifact(metadata_path),
        results_path.name: _artifact(results_path),
        evidence_path.name: _artifact(evidence_path),
        validation_path.name: {
            "filename": validation_path.name,
            "sha256": None,
            "digest_type": "canonical-json-sha256",
            "sha256_scope": (
                "canonical validation payload with this sha256 field set to null; "
                "all other artifact digests are included"
            ),
        },
    }
    validation_payload: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "warning" if warnings or provider_errors else "ok",
        "artifacts": artifacts,
        "software_versions": _software_versions(),
        "provider_calls": provider_calls,
        "record_counts": record_counts(records),
        "missing_inputs": _missing_inputs(frame),
        "channel_capabilities": capabilities,
        "quality_warnings": warnings,
        "schema_validation": {
            "draft": "2020-12",
            "analysis_results": {"valid": True, "errors": []},
            "analysis_metadata": {"valid": True, "errors": []},
            "analysis_evidence": {"valid": True, "errors": []},
            "analysis_validation": {"valid": True, "errors": []},
        },
        "recomputation": {
            "input_report_equals_provider_payload": True,
            "input_adapter": handoff.adapter,
            "input_provider": handoff.provider,
            "input_provider_version": handoff.provider_version,
            "input_report_file": report_path.name,
            "handoff_conformance_status": handoff.conformance_status,
            "bds_schema_version": (
                direct_report.get("schema_version") if handoff.adapter == "bds" else None
            ),
            "bds_target": "legacy_bdf_style" if handoff.adapter == "bds" else None,
            "bdf_validator_passed": (
                bool(direct_report.get("ok")) if handoff.adapter == "bdf" else None
            ),
            "bdf_schema_version": {
                "value": None,
                "status": (
                    "not_reported_by_batterydf_validator"
                    if handoff.adapter == "bdf"
                    else "not_reported_by_bds_legacy_export"
                ),
                "reason": "provider package version is not a formal BDF ontology version",
            },
            "normalized_row_count": analysis_frame_details["normalized_row_count"],
            "analysis_row_count": analysis_frame_details["analysis_row_count"],
            "pyprobe_bridge_row_count": pyprobe_frame.height if pyprobe_frame is not None else 0,
            "row_count_preserved": (
                analysis_frame_details["analysis_row_count"] == pyprobe_frame.height
                if pyprobe_frame is not None
                else None
            ),
            "excluded_rows_without_finite_time": analysis_frame_details[
                "excluded_rows_without_finite_time"
            ],
            "excluded_source_rows_without_finite_time": analysis_frame_details[
                "excluded_source_rows_without_finite_time"
            ],
            "capacity_method": adapter_details["capacity_method"],
            "capacity_flags": adapter_details["capacity_flags"],
            "current_sign": (
                direct_report.get("current_sign") if handoff.adapter == "bds" else "charge-positive"
            ),
            "bds_time_sampling": (
                direct_report.get("metadata", {}).get("time_sampling", {})
                if handoff.adapter == "bds"
                else {}
            ),
            "bds_semantic_sources": (
                direct_report.get("metadata", {}).get("semantic_sources", {})
                if handoff.adapter == "bds"
                else {}
            ),
            "analysis_channels": channel_details,
            "compact_summary_evidence_references_valid": all(
                item["evidence"]["record_id"] in set(record_ids)
                for dimension in results_payload["dimensions"].values()
                for item in dimension
            ),
            **_record_recomputation(records, frame),
        },
    }
    validation_payload["artifacts"][validation_path.name]["sha256"] = canonical_sha256(
        validation_payload
    )
    validation_errors = validate_payload(validation_payload, VALIDATION_SCHEMA)
    if validation_errors:
        raise ValueError("analysis_validation.json schema errors: " + "; ".join(validation_errors))
    write_json(validation_path, validation_payload)
    return AnalysisResult(
        output_dir=output_dir,
        normalized_data_path=normalized_path,
        input_report_path=report_path,
        analysis_results_path=results_path,
        analysis_metadata_path=metadata_path,
        analysis_evidence_path=evidence_path,
        analysis_validation_path=validation_path,
        records=tuple(records),
    )


def _prepare_output_directory(output_dir: Path, report_filename: str) -> None:
    """Refuse to mix a six-file run with unrelated or stale artifacts."""

    expected = {
        "normalized_data.bdf.parquet",
        report_filename,
        "analysis_metadata.json",
        "analysis_results.json",
        "analysis_evidence.json",
        "analysis_validation.json",
    }
    if output_dir.exists():
        extras = sorted(path.name for path in output_dir.iterdir() if path.name not in expected)
        if extras:
            raise FileExistsError(
                "output_dir contains files outside this run contract; choose a clean directory: "
                f"{extras}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


def _frame_interval(frame: polars.DataFrame) -> dict[str, Any] | None:
    """Return the exact complete observation interval when it can be located."""

    if frame.is_empty() or "_source_row" not in frame.columns:
        return None
    ordered = frame.sort("test_time_s")
    rows = ordered["_source_row"].cast(polars.Int64, strict=False).to_list()
    times = ordered["test_time_s"].cast(polars.Float64, strict=False).to_list()
    interval: dict[str, Any] = {
        "start_row": int(rows[0]),
        "end_row": int(rows[-1]),
        "start_time_s": float(times[0]),
        "end_time_s": float(times[-1]),
    }
    if "record_index" in ordered.columns:
        records = ordered["record_index"].cast(polars.Int64, strict=False).to_list()
        if records[0] is not None and records[-1] is not None:
            interval["start_record"] = int(records[0])
            interval["end_record"] = int(records[-1])
    return interval


def _missing_channel_record(
    record_type: str,
    cell_id: str,
    *,
    reason: str,
    source_interval: dict[str, Any] | None,
    provider: str,
    method: str,
) -> dict[str, Any]:
    """Create a stable typed record when a feature lacks a required channel."""

    return make_record(
        record_id=f"{record_type}:None:unavailable",
        record_type=record_type,
        cell_id=cell_id,
        cycle_scope=None,
        source_intervals=[source_interval] if source_interval is not None else [],
        attributes={
            "required_channels": ["test_time_s", "current_a", "voltage_v"],
            "reference_frame": {"scope": "complete_observation_window"},
        },
        metrics={"availability": metric(None, "1", status="not_computable", reason=reason)},
        provider=provider,
        method_name=method,
        provider_version="not_invoked",
        parameters={},
        references=[],
        applicability_status="not_computable",
        applicability_reasons=[reason],
        quality_status="warning",
        quality_flags=[reason],
        interpretation_limits=[
            "This analysis was not run because a required standardized measurement channel is unavailable."
        ],
    )


def _select_representative_cycle(
    config: AnalysisConfig, summaries: list[dict[str, Any]]
) -> tuple[int | None, str]:
    """Select a cycle for analyses that explicitly require a cycle reference frame.

    Early-cycle exclusion is treated as configuration only. This helper does not
    infer that an excluded cycle is a formation cycle.
    """

    if config.representative_cycle is not None:
        available = {item["cycle_id"] for item in summaries}
        if config.representative_cycle not in available:
            raise ValueError(
                f"representative_cycle {config.representative_cycle} is not present in BDS output"
            )
        return config.representative_cycle, "explicit"

    ordered = sorted(summaries, key=lambda item: item["cycle_id"])
    exclude_count = max(0, int(config.formation_cycles_to_exclude))
    excluded = {item["cycle_id"] for item in ordered[:exclude_count]}
    complete = [
        item for item in ordered if item.get("complete") and item["cycle_id"] not in excluded
    ]
    if complete:
        return (
            int(complete[0]["cycle_id"]),
            "first_complete_after_configured_early_cycle_exclusion",
        )
    return None, "no_complete_cycle_after_configured_early_cycle_exclusion"


def _configuration_dict(config: AnalysisConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["analysis_policy_version"] = ANALYSIS_POLICY_VERSION
    payload["output_dir"] = str(config.output_dir)
    payload["pulse_resistance_times_s"] = list(config.pulse_resistance_times_s)
    payload["relaxation_checkpoints_s"] = list(config.relaxation_checkpoints_s)
    return payload


def _run_id(source: Path, config: AnalysisConfig) -> str:
    identity_configuration = _configuration_dict(config)
    identity_configuration.pop("output_dir", None)
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(json.dumps(identity_configuration, sort_keys=True).encode("utf-8"))
    return f"bfl-{digest.hexdigest()[:16]}"


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "battery-feature-lab": importlib.metadata.version("battery-feature-lab"),
        "battery-data-standard": importlib.metadata.version("battery-data-standard"),
        "PyProBE-Data": importlib.metadata.version("PyProBE-Data"),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "polars": polars.__version__,
        "bds_module": getattr(bds, "__version__", "unknown"),
        "pyprobe_module": getattr(pyprobe, "__version__", "unknown"),
    }
    try:
        versions["batterydf"] = importlib.metadata.version("batterydf")
    except importlib.metadata.PackageNotFoundError:
        versions["batterydf"] = "not-installed"
    return versions


def _missing_inputs(frame: Any) -> list[str]:
    """Report only globally required canonical inputs that are unavailable.

    Nominal capacity, protocol name, temperature, and source cycle identifiers
    are contextual or feature-specific inputs. Their absence should gate only
    the analyses that require them, rather than being reported as a global
    missing-input condition.
    """

    required = ("test_time_s", "current_a")
    columns = set(getattr(frame, "columns", []))
    return [name for name in required if name not in columns]


def _metric_value(record: dict[str, Any], name: str) -> float | None:
    """Return a finite numeric metric value, or None when unavailable."""

    item = record.get("metrics", {}).get(name, {})
    value = item.get("value") if isinstance(item, dict) else None
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numpy.isfinite(numeric) else None


def _record_recomputation(
    records: list[dict[str, Any]], frame: polars.DataFrame
) -> dict[str, Any]:
    """Recompute selected identities without assuming every record is computable."""

    exposure = next(
        (record for record in records if record.get("record_type") == "operation.exposure_summary"),
        None,
    )
    excluded_duration = (
        _metric_value(exposure, "excluded_duration") if exposure is not None else None
    )
    interval_outlier_flags = sum(
        "sampling_interval_outlier" in record.get("quality", {}).get("flags", [])
        for record in records
    )
    cumulative_flags = sorted(
        {
            flag
            for record in records
            for flag in record.get("quality", {}).get("flags", [])
            if flag.startswith("reported_")
        }
    )

    # Cycle-summary identities are checked only when the required metrics are
    # present and finite. A not-computable cycle summary is valid output and
    # must not make validation itself fail.
    coulombic_errors: list[float] = []
    energy_errors: list[float] = []
    for record in records:
        if record.get("record_type") != "response.cycle_summary":
            continue

        charge = _metric_value(record, "charge_capacity")
        discharge = _metric_value(record, "discharge_capacity")
        efficiency = _metric_value(record, "coulombic_efficiency")
        if charge is not None and charge > 0 and discharge is not None and efficiency is not None:
            coulombic_errors.append(abs(efficiency - discharge / charge))

        charge_energy = _metric_value(record, "charge_energy")
        discharge_energy = _metric_value(record, "discharge_energy")
        energy_efficiency = _metric_value(record, "energy_efficiency")
        if (
            charge_energy is not None
            and charge_energy > 0
            and discharge_energy is not None
            and energy_efficiency is not None
        ):
            energy_errors.append(abs(energy_efficiency - discharge_energy / charge_energy))

    current_step_errors: list[float] = []
    for record in records:
        if (
            record.get("record_type") != "response.current_step"
            or record.get("applicability", {}).get("status") != "applicable"
        ):
            continue

        metrics = record.get("metrics", {})
        resistance_names = [name for name in metrics if name.startswith("apparent_dc_resistance_")]
        for resistance_name in resistance_names:
            suffix = resistance_name.removeprefix("apparent_dc_resistance_")
            voltage_name = f"delta_voltage_{suffix}"
            current_name = f"delta_current_{suffix}"
            resistance = _metric_value(record, resistance_name)
            voltage_delta = _metric_value(record, voltage_name)
            current_delta = _metric_value(record, current_name)
            if (
                resistance is not None
                and voltage_delta is not None
                and current_delta is not None
                and current_delta != 0
            ):
                current_step_errors.append(abs(resistance - voltage_delta / current_delta))

    directional_errors: list[float] = []
    for record in records:
        if record.get("record_type") != "response.directional_energy_summary":
            continue

        charge_ah = _metric_value(record, "charge_throughput")
        discharge_ah = _metric_value(record, "discharge_throughput")
        charge_wh = _metric_value(record, "charge_energy")
        discharge_wh = _metric_value(record, "discharge_energy")
        charge_voltage = _metric_value(record, "charge_mean_voltage")
        discharge_voltage = _metric_value(record, "discharge_mean_voltage")
        voltage_gap = _metric_value(record, "directional_mean_voltage_gap")
        balance = _metric_value(record, "charge_discharge_throughput_balance")
        energy_return = _metric_value(record, "balanced_window_energy_return_ratio")

        if (
            charge_ah is not None
            and charge_ah > 0
            and charge_wh is not None
            and charge_voltage is not None
        ):
            directional_errors.append(abs(charge_voltage - charge_wh / charge_ah))

        if (
            discharge_ah is not None
            and discharge_ah > 0
            and discharge_wh is not None
            and discharge_voltage is not None
        ):
            directional_errors.append(abs(discharge_voltage - discharge_wh / discharge_ah))

        if charge_voltage is not None and discharge_voltage is not None and voltage_gap is not None:
            directional_errors.append(abs(voltage_gap - (charge_voltage - discharge_voltage)))

        if charge_ah is not None and discharge_ah is not None and balance is not None:
            maximum = max(charge_ah, discharge_ah)
            if maximum > 0:
                directional_errors.append(abs(balance - min(charge_ah, discharge_ah) / maximum))

        if (
            charge_wh is not None
            and charge_wh > 0
            and discharge_wh is not None
            and energy_return is not None
        ):
            directional_errors.append(abs(energy_return - discharge_wh / charge_wh))

    summary_count_errors: list[float] = []
    steps_by_scope: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("record_type") == "response.current_step":
            steps_by_scope.setdefault(record.get("cycle_scope"), []).append(record)

    for record in records:
        if record.get("record_type") != "response.current_step_summary":
            continue

        steps = steps_by_scope.get(record.get("cycle_scope"), [])
        candidates = [
            step for step in steps if step.get("attributes", {}).get("candidate_detected", True)
        ]
        computed = [
            step
            for step in candidates
            if step.get("applicability", {}).get("status") == "applicable"
        ]
        expected = {
            "candidate_step_count": len(candidates),
            "computed_step_count": len(computed),
            "rejected_step_count": len(candidates) - len(computed),
        }
        for name, value in expected.items():
            observed = _metric_value(record, name)
            if observed is not None:
                summary_count_errors.append(abs(observed - value))

    relaxation_errors: list[float] = []
    relaxation_absolute_errors: list[float] = []
    relaxation_checkpoint_errors: list[float] = []
    rests_by_scope: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        if (
            record.get("record_type") == "response.rest_and_thermal"
            and record.get("applicability", {}).get("status") == "applicable"
        ):
            rests_by_scope.setdefault(record.get("cycle_scope"), []).append(record)

    for record in records:
        if record.get("record_type") != "response.relaxation_signature":
            continue

        rests = rests_by_scope.get(record.get("cycle_scope"), [])
        checkpoints = record.get("method", {}).get("parameters", {}).get("checkpoints_s", [])
        for checkpoint in checkpoints:
            try:
                checkpoint_value = float(checkpoint)
            except (TypeError, ValueError):
                continue
            if not numpy.isfinite(checkpoint_value) or checkpoint_value <= 0:
                continue

            suffix = seconds_label(checkpoint_value)
            recoveries: list[float] = []
            absolute_changes: list[float] = []
            name = f"voltage_change_at_{suffix}s"
            for rest in rests:
                item = rest.get("metrics", {}).get(name, {})
                delta = (
                    item.get("value")
                    if isinstance(item, dict) and item.get("status") == "ok"
                    else None
                )
                try:
                    delta_value = float(delta) if delta is not None else None
                except (TypeError, ValueError):
                    delta_value = None
                if delta_value is None or not numpy.isfinite(delta_value):
                    continue
                absolute_changes.append(abs(delta_value))

                previous = rest.get("attributes", {}).get("previous_phase")
                if previous == "charge":
                    recoveries.append(-delta_value)
                elif previous == "discharge":
                    recoveries.append(delta_value)

            observed = _metric_value(record, f"polarization_recovery_q50_{suffix}s")
            if len(recoveries) >= 3 and observed is not None:
                relaxation_errors.append(abs(observed - float(numpy.quantile(recoveries, 0.5))))
            absolute_observed = _metric_value(record, f"absolute_voltage_change_q50_{suffix}s")
            if len(absolute_changes) >= 3 and absolute_observed is not None:
                relaxation_absolute_errors.append(
                    abs(absolute_observed - float(numpy.quantile(absolute_changes, 0.5)))
                )

    for record in records:
        if (
            record.get("record_type") != "response.rest_and_thermal"
            or record.get("applicability", {}).get("status") != "applicable"
            or not record.get("source_intervals")
        ):
            continue
        interval = record["source_intervals"][0]
        start_row = interval.get("start_row")
        end_row = interval.get("end_row")
        if start_row is None or end_row is None:
            continue
        source = frame.filter(
            (polars.col("_source_row") >= int(start_row))
            & (polars.col("_source_row") <= int(end_row))
        ).sort("test_time_s")
        if source.height < 2 or "voltage_v" not in source.columns:
            continue
        time = source["test_time_s"].cast(polars.Float64, strict=False).to_numpy()
        voltage = source["voltage_v"].cast(polars.Float64, strict=False).to_numpy()
        finite = numpy.isfinite(time) & numpy.isfinite(voltage)
        time = time[finite]
        voltage = voltage[finite]
        if len(time) < 2:
            continue
        relative = time - time[0]
        checkpoints = record.get("method", {}).get("parameters", {}).get(
            "checkpoints_s", []
        )
        for checkpoint in checkpoints:
            checkpoint_value = float(checkpoint)
            if checkpoint_value < relative[0] or checkpoint_value > relative[-1]:
                continue
            suffix = seconds_label(checkpoint_value)
            expected_voltage = float(numpy.interp(checkpoint_value, relative, voltage))
            observed_voltage = _metric_value(record, f"voltage_at_{suffix}s")
            observed_change = _metric_value(record, f"voltage_change_at_{suffix}s")
            if observed_voltage is not None:
                relaxation_checkpoint_errors.append(
                    abs(observed_voltage - expected_voltage)
                )
            if observed_change is not None:
                relaxation_checkpoint_errors.append(
                    abs(observed_change - (expected_voltage - float(voltage[0])))
                )

    capacity_profile_errors: list[float] = []
    capacity_profile_series_errors: list[float] = []
    for record in records:
        if record.get("record_type") != "response.capacity_aligned_profile":
            continue
        charge = _metric_value(record, "charge_capacity")
        discharge = _metric_value(record, "discharge_capacity")
        balance = _metric_value(record, "directional_capacity_balance")
        common = _metric_value(record, "aligned_capacity")
        if charge is not None and discharge is not None:
            maximum = max(charge, discharge)
            if balance is not None and maximum > 0:
                capacity_profile_errors.append(abs(balance - min(charge, discharge) / maximum))
            if common is not None:
                capacity_profile_errors.append(abs(common - min(charge, discharge)))
        candidate_charge = _metric_value(record, "candidate_charge_capacity")
        candidate_discharge = _metric_value(record, "candidate_discharge_capacity")
        candidate_balance = _metric_value(record, "candidate_capacity_balance")
        candidate_unpaired = _metric_value(record, "candidate_unpaired_capacity")
        if candidate_charge is not None and candidate_discharge is not None:
            candidate_maximum = max(candidate_charge, candidate_discharge)
            if candidate_balance is not None and candidate_maximum > 0:
                capacity_profile_errors.append(
                    abs(
                        candidate_balance
                        - min(candidate_charge, candidate_discharge) / candidate_maximum
                    )
                )
            if candidate_unpaired is not None:
                capacity_profile_errors.append(
                    abs(candidate_unpaired - abs(candidate_charge - candidate_discharge))
                )
        series = record.get("series", {})
        voltage_charge = series.get("charge_voltage_v", [])
        voltage_discharge = series.get("discharge_voltage_v", [])
        voltage_gap = series.get("voltage_gap_v", [])
        capacity_axis = series.get("capacity_from_shared_upper_endpoint_ah", [])
        lengths = {
            len(voltage_charge),
            len(voltage_discharge),
            len(voltage_gap),
            len(capacity_axis),
        }
        if lengths != {0} and len(lengths) == 1:
            for charge_v, discharge_v, gap_v in zip(
                voltage_charge, voltage_discharge, voltage_gap, strict=True
            ):
                capacity_profile_series_errors.append(
                    abs(float(gap_v) - (float(charge_v) - float(discharge_v)))
                )
            if len(capacity_axis) >= 2:
                axis = numpy.asarray(capacity_axis, dtype=float)
                capacity_profile_series_errors.append(
                    0.0 if numpy.all(numpy.diff(axis) > 0) else 1.0
                )
        elif lengths != {0}:
            capacity_profile_series_errors.append(1.0)

    return {
        "record_ids_unique": len({record["record_id"] for record in records}) == len(records),
        # Kept for validation-schema compatibility. The referenced exposure
        # record now describes the default observation scope.
        "representative_excluded_duration_s": excluded_duration,
        "records_with_sampling_interval_outlier": interval_outlier_flags,
        "reported_cumulative_column_flags": cumulative_flags,
        "efficiency_recomputation": {
            "coulombic_cycles_checked": len(coulombic_errors),
            "coulombic_max_absolute_error": max(coulombic_errors, default=None),
            "energy_cycles_checked": len(energy_errors),
            "energy_max_absolute_error": max(energy_errors, default=None),
        },
        "short_window_recomputation": {
            "current_step_ratios_checked": len(current_step_errors),
            "current_step_max_absolute_error_ohm": max(current_step_errors, default=None),
            "directional_identities_checked": len(directional_errors),
            "directional_max_absolute_error": max(directional_errors, default=None),
            "current_step_summary_counts_checked": len(summary_count_errors),
            "current_step_summary_count_max_absolute_error": max(
                summary_count_errors, default=None
            ),
            "relaxation_medians_checked": len(relaxation_errors),
            "relaxation_median_max_absolute_error_v": max(relaxation_errors, default=None),
            "relaxation_absolute_medians_checked": len(relaxation_absolute_errors),
            "relaxation_absolute_median_max_absolute_error_v": max(
                relaxation_absolute_errors, default=None
            ),
            "relaxation_checkpoint_identities_checked": len(
                relaxation_checkpoint_errors
            ),
            "relaxation_checkpoint_max_absolute_error_v": max(
                relaxation_checkpoint_errors, default=None
            ),
            "capacity_profile_identities_checked": len(capacity_profile_errors),
            "capacity_profile_max_absolute_error": max(capacity_profile_errors, default=None),
            "capacity_profile_series_identities_checked": len(capacity_profile_series_errors),
            "capacity_profile_series_max_absolute_error": max(
                capacity_profile_series_errors, default=None
            ),
        },
    }
