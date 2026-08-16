"""Compact AI-facing index over the full auditable evidence records."""

from __future__ import annotations

from typing import Any

CORE_TYPES = {
    "operation.window_summary",
    "response.directional_energy_summary",
    "response.capacity_aligned_profile",
    "response.current_step_summary",
    "response.pulse_resistance",
    "response.relaxation_signature",
    "response.cycle_summary",
    "response.ica_curve",
    "response.dva_curve",
    "evolution.capacity",
}

KEY_METRICS: dict[str, set[str]] = {
    "operation.window_summary": {
        "total_duration",
        "rest_fraction",
        "charge_fraction",
        "discharge_fraction",
        "capacity_throughput",
        "charge_throughput",
        "discharge_throughput",
        "energy_throughput",
        "absolute_power_q95",
        "current_squared_exposure",
        "absolute_current_q50",
        "absolute_current_q95",
        "voltage_q05",
        "voltage_q50",
        "voltage_q95",
        "voltage_min",
        "voltage_max",
        "temperature_min",
        "temperature_q50",
        "temperature_q95",
        "temperature_max",
        "temperature_duration_coverage",
        "constant_current_like_active_fraction",
        "constant_voltage_like_active_fraction",
        "pulse_like_active_fraction",
        "dynamic_current_active_fraction",
        "classified_active_fraction",
    },
    "response.directional_energy_summary": {
        "charge_throughput",
        "discharge_throughput",
        "charge_energy",
        "discharge_energy",
        "charge_mean_voltage",
        "discharge_mean_voltage",
        "directional_mean_voltage_gap",
        "charge_discharge_throughput_balance",
        "balanced_window_energy_return_ratio",
    },
    "response.current_step_summary": {
        "candidate_step_count",
        "computed_step_count",
        "rejected_step_count",
        "absolute_delta_current_q50",
        "apparent_dc_resistance_first_valid_q50",
        "apparent_dc_resistance_2s_q50",
        "apparent_dc_resistance_10s_q50",
        "pre_step_voltage_q05",
        "pre_step_voltage_q95",
        "pre_step_temperature_q05",
        "pre_step_temperature_q95",
    },
}


def _series_index(series: dict[str, Any]) -> dict[str, Any]:
    lengths = {name: len(values) for name, values in series.items() if isinstance(values, list)}
    return {
        "available": bool(series),
        "keys": sorted(series),
        "point_counts": lengths,
    }


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    attributes = record.get("attributes", {})
    keep_attributes = {
        name: attributes[name]
        for name in (
            "analysis_context",
            "confidence",
            "contributing_counts_by_preceding_mode",
            "contributing_counts_by_previous_phase",
            "direction_counts",
            "dominant_active_mode",
            "mode_segment_counts",
            "operation_sequence",
            "operation_sequence_status",
            "operation_sequence_summary_limit",
            "pairing",
            "phase_segment_counts",
            "reference_frame",
            "rejection_counts_by_reason",
        )
        if name in attributes
    }
    member_ids = attributes.get("member_record_ids", [])
    member_summaries = attributes.get("member_summaries", [])
    member_limit = 32
    if isinstance(member_ids, list):
        keep_attributes["member_record_count"] = len(member_ids)
        keep_attributes["member_record_ids"] = member_ids if len(member_ids) <= member_limit else []
    if isinstance(member_summaries, list):
        keep_attributes["member_summaries"] = (
            member_summaries if len(member_summaries) <= member_limit else []
        )
        keep_attributes["member_summaries_status"] = (
            "included" if len(member_summaries) <= member_limit else "retrieve_aggregate_evidence"
        )
    metrics = record.get("metrics", {})
    selected_names = KEY_METRICS.get(record["record_type"])
    if selected_names is not None:
        metrics = {name: value for name, value in metrics.items() if name in selected_names}
    elif record["record_type"] == "response.relaxation_signature":
        metrics = {
            name: value
            for name, value in metrics.items()
            if name.startswith(
                (
                    "eligible_rest_count_",
                    "absolute_voltage_change_q50_",
                    "absolute_voltage_change_q95_",
                    "polarization_recovery_q50_",
                )
            )
            or name
            in {
                "rest_segment_count",
                "phase_conditioned_rest_segment_count",
                "unconditioned_rest_segment_count",
                "rejected_rest_segment_count",
            }
        }
    return {
        "record_id": record["record_id"],
        "record_type": record["record_type"],
        "cycle_scope": record.get("cycle_scope"),
        "attributes": keep_attributes,
        "metrics": metrics,
        "series": _series_index(record.get("series", {})),
        "applicability": record.get("applicability", {}),
        "quality": record.get("quality", {}),
        "interpretation_limits": record.get("interpretation_limits", []),
        "evidence": {
            "file": "analysis_evidence.json",
            "record_id": record["record_id"],
            "source_interval_count": len(record.get("source_intervals", [])),
        },
    }


def compile_summary(records: list[dict[str, Any]], *, run_id: str, cell_id: str) -> dict[str, Any]:
    """Create a small index; detailed arrays and provenance stay in evidence."""

    dimensions: dict[str, list[dict[str, Any]]] = {
        "operation": [],
        "response": [],
        "evolution": [],
    }
    record_type_counts: dict[str, int] = {}
    for record in records:
        record_type = record.get("record_type", "")
        record_type_counts[record_type] = record_type_counts.get(record_type, 0) + 1
        if record_type not in CORE_TYPES:
            continue
        dimension = record_type.split(".", 1)[0]
        dimensions[dimension].append(_compact_record(record))
    return {
        "schema_version": "bfl.summary/0.1",
        "run_id": run_id,
        "cell_id": cell_id,
        "metadata_file": "analysis_metadata.json",
        "evidence_file": "analysis_evidence.json",
        "validation_file": "analysis_validation.json",
        "dimensions": dimensions,
        "retrieval": {
            "join_key": "record_id",
            "instructions": (
                "Read this file first. Resolve a record_id in analysis_evidence.json "
                "only when source intervals, derivation details, parameters, references, "
                "or full series are needed."
            ),
            "evidence_record_count": len(records),
            "record_type_counts": dict(sorted(record_type_counts.items())),
        },
    }
