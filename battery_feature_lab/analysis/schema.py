"""Public configuration, result objects, and JSON schemas for BFL analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = "bfl.analysis/0.1"
SUMMARY_SCHEMA_VERSION = "bfl.summary/0.1"
METADATA_SCHEMA_VERSION = "bfl.metadata/0.1"
VALIDATION_SCHEMA_VERSION = "bfl.validation/0.1"
ANALYSIS_POLICY_VERSION = "bfl.analysis-policy/0.1"
DEFAULT_RELAXATION_CHECKPOINTS_S = (10.0, 30.0, 60.0, 300.0, 600.0, 1800.0)


def seconds_label(value: float) -> str:
    """Return a stable, lossless-enough JSON key label for a time in seconds."""

    return format(float(value), ".15g")

DEFAULT_ANALYSIS_POLICY: Mapping[str, float] = MappingProxyType({
    "mode_cc_current_cv_max": 0.15,
    "mode_cv_voltage_range_max_v": 0.02,
    "mode_cv_voltage_slope_max_v_per_s": 2e-4,
    "mode_cv_taper_ratio_max": 0.8,
    "mode_cv_nonincreasing_fraction_min": 0.8,
    "mode_min_samples": 3.0,
    "mode_taper_absolute_noise_a": 1e-5,
    "mode_taper_relative_noise_fraction": 0.001,
    "mode_pulse_max_duration_s": 60.0,
    "mode_adjacent_rest_min_s": 60.0,
    "mode_dominant_classified_fraction_min": 0.5,
    "directional_balance_min": 0.95,
    "current_step_rest_min_s": 60.0,
    "current_step_pulse_max_s": 60.0,
    "current_step_baseline_window_s": 30.0,
    "current_step_min_samples": 3.0,
    "current_step_min_delta_current_a": 0.01,
    "current_step_delta_current_noise_multiplier": 10.0,
    "current_step_baseline_rest_fraction_min": 0.95,
    "current_step_plateau_cv_max": 0.05,
    "current_step_plateau_match_relative_tolerance": 0.05,
    "current_step_high_confidence_cv_max": 0.02,
    "current_step_summary_min_count": 3.0,
    "relaxation_summary_min_count": 3.0,
    "profile_min_samples": 50.0,
    "profile_min_duration_s": 600.0,
    "profile_finite_coverage_min": 0.99,
    "profile_direction_consistency_min": 0.99,
    "profile_bracketing_rest_min_s": 60.0,
    "profile_grid_points": 101.0,
    "profile_endpoint_tolerance_v": 0.15,
    "profile_capacity_balance_min": 0.95,
    "profile_voltage_span_overlap_min": 0.9,
    "profile_current_ratio_min": 0.8,
    "profile_current_ratio_max": 1.25,
    "profile_current_match_fraction_min": 0.8,
    "profile_voltage_wrong_way_fraction_max": 0.05,
    "profile_temperature_difference_max_deg_c": 2.0,
    "diagnostic_min_samples": 50.0,
    "diagnostic_finite_coverage_min": 0.99,
    "diagnostic_monotone_fraction_min": 0.99,
    "diagnostic_min_voltage_span_v": 0.5,
    "diagnostic_max_c_rate": 0.2,
    "diagnostic_peak_prominence_fraction": 0.05,
    "diagnostic_peak_minimum_distance_points": 3.0,
    "cycle_integration_coverage_min": 0.99,
    "cycle_efficiency_warning_max": 1.05,
    "reported_capacity_finite_coverage_min": 0.99,
    "metadata_finite_coverage_warning_min": 0.99,
    "metadata_quantization_min_samples": 100.0,
    "metadata_quantization_unique_fraction_max": 0.01,
    "metadata_quantization_relative_increment_epsilon": 1e-9,
    "metadata_voltage_quantization_increment_min_v": 0.001,
    "metadata_nonvoltage_quantization_increment_min": 0.01,
})


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for one deterministic Battery Feature Lab run."""

    output_dir: Path = Path("bfl_outputs")
    input_adapter: str = "auto"
    cell_id: str | None = None
    nominal_capacity_ah: float | None = None
    representative_cycle: int | None = None
    declared_protocol_name: str | None = None
    formation_cycles_to_exclude: int = 1
    reference_window_size: int = 4
    pulse_resistance_times_s: tuple[float, ...] = (10.0,)
    relaxation_checkpoints_s: tuple[float, ...] = DEFAULT_RELAXATION_CHECKPOINTS_S
    voltage_column: str | None = None
    temperature_column: str | None = None
    rest_current_threshold_a: float = 1e-4
    sampling_interval_outlier_factor: float = 5.0
    analysis_policy: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        supplied_policy: Mapping[str, Any] = self.analysis_policy
        unknown_policy = sorted(set(supplied_policy) - set(DEFAULT_ANALYSIS_POLICY))
        if unknown_policy:
            raise ValueError(f"unknown analysis_policy keys: {unknown_policy}")
        policy = {**DEFAULT_ANALYSIS_POLICY, **dict(supplied_policy)}
        for name, value in policy.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"analysis_policy value must be numeric: {name}") from exc
            if not math.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"analysis_policy value must be finite and positive: {name}")
            policy[name] = numeric
        for integer_name in (
            "profile_min_samples",
            "profile_grid_points",
            "diagnostic_min_samples",
            "diagnostic_peak_minimum_distance_points",
            "mode_min_samples",
            "current_step_min_samples",
            "current_step_summary_min_count",
            "relaxation_summary_min_count",
            "metadata_quantization_min_samples",
        ):
            if not policy[integer_name].is_integer():
                raise ValueError(f"analysis_policy value must be an integer: {integer_name}")
        unit_interval_names = (
            "mode_cc_current_cv_max",
            "mode_cv_taper_ratio_max",
            "mode_cv_nonincreasing_fraction_min",
            "mode_taper_relative_noise_fraction",
            "mode_dominant_classified_fraction_min",
            "directional_balance_min",
            "current_step_baseline_rest_fraction_min",
            "current_step_plateau_cv_max",
            "current_step_plateau_match_relative_tolerance",
            "current_step_high_confidence_cv_max",
            "profile_finite_coverage_min",
            "profile_direction_consistency_min",
            "profile_capacity_balance_min",
            "profile_voltage_span_overlap_min",
            "profile_current_ratio_min",
            "profile_current_match_fraction_min",
            "profile_voltage_wrong_way_fraction_max",
            "diagnostic_finite_coverage_min",
            "diagnostic_monotone_fraction_min",
            "diagnostic_peak_prominence_fraction",
            "reported_capacity_finite_coverage_min",
            "metadata_finite_coverage_warning_min",
            "metadata_quantization_unique_fraction_max",
        )
        for fraction_name in unit_interval_names:
            if policy[fraction_name] > 1:
                raise ValueError(
                    f"analysis_policy fraction must be at most one: {fraction_name}"
                )
        if policy["profile_current_ratio_min"] >= policy["profile_current_ratio_max"]:
            raise ValueError(
                "analysis_policy profile_current_ratio_min must be less than "
                "profile_current_ratio_max"
            )
        if policy["profile_grid_points"] < 3:
            raise ValueError("analysis_policy profile_grid_points must be at least three")
        object.__setattr__(self, "analysis_policy", policy)
        if self.input_adapter not in {"auto", "bds", "bdf"}:
            raise ValueError("input_adapter must be one of: auto, bds, bdf")
        object.__setattr__(
            self,
            "pulse_resistance_times_s",
            tuple(sorted({float(value) for value in self.pulse_resistance_times_s})),
        )
        object.__setattr__(
            self,
            "relaxation_checkpoints_s",
            tuple(sorted({float(value) for value in self.relaxation_checkpoints_s})),
        )
        if self.nominal_capacity_ah is not None and (
            not math.isfinite(float(self.nominal_capacity_ah))
            or self.nominal_capacity_ah <= 0
        ):
            raise ValueError("nominal_capacity_ah must be finite and positive")
        if self.formation_cycles_to_exclude < 0:
            raise ValueError("formation_cycles_to_exclude must be non-negative")
        if self.reference_window_size < 1:
            raise ValueError("reference_window_size must be at least one")
        if not self.pulse_resistance_times_s or any(
            not math.isfinite(value) or value <= 0
            for value in self.pulse_resistance_times_s
        ):
            raise ValueError("pulse_resistance_times_s values must be finite and positive")
        if not self.relaxation_checkpoints_s or any(
            not math.isfinite(value) or value <= 0
            for value in self.relaxation_checkpoints_s
        ):
            raise ValueError("relaxation_checkpoints_s values must be finite and positive")
        pulse_labels = [seconds_label(value) for value in self.pulse_resistance_times_s]
        if len(pulse_labels) != len(set(pulse_labels)):
            raise ValueError("pulse_resistance_times_s values must have unique key labels")
        checkpoint_labels = [seconds_label(value) for value in self.relaxation_checkpoints_s]
        if len(checkpoint_labels) != len(set(checkpoint_labels)):
            raise ValueError("relaxation_checkpoints_s values must have unique key labels")
        for name, value in (
            ("voltage_column", self.voltage_column),
            ("temperature_column", self.temperature_column),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{name} must be a non-empty exact column name")
        if (
            not math.isfinite(float(self.rest_current_threshold_a))
            or self.rest_current_threshold_a < 0
        ):
            raise ValueError("rest_current_threshold_a must be finite and non-negative")
        if (
            not math.isfinite(float(self.sampling_interval_outlier_factor))
            or self.sampling_interval_outlier_factor <= 1
        ):
            raise ValueError(
                "sampling_interval_outlier_factor must be finite and greater than one"
            )

    def policy(self, name: str) -> float:
        """Return one validated, versioned default or caller override."""

        return self.analysis_policy[name]


@dataclass(frozen=True)
class AnalysisResult:
    """Paths and records produced by :func:`battery_feature_lab.analyze`."""

    output_dir: Path
    normalized_data_path: Path
    input_report_path: Path
    analysis_results_path: Path
    analysis_metadata_path: Path
    analysis_evidence_path: Path
    analysis_validation_path: Path
    records: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def files(self) -> tuple[Path, ...]:
        """Return the stable machine-readable output files in contract order."""

        return (
            self.normalized_data_path,
            self.input_report_path,
            self.analysis_metadata_path,
            self.analysis_results_path,
            self.analysis_evidence_path,
            self.analysis_validation_path,
        )

    @property
    def bds_conversion_report_path(self) -> Path | None:
        """Return the native BDS report path for a BDS-backed run."""

        return (
            self.input_report_path
            if self.input_report_path.name == "bds_conversion_report.json"
            else None
        )

    @property
    def bdf_validation_report_path(self) -> Path | None:
        """Return the native formal-BDF validation report path when used."""

        return (
            self.input_report_path
            if self.input_report_path.name == "bdf_validation_report.json"
            else None
        )


def metric(
    value: float | None,
    unit: str,
    *,
    status: str = "ok",
    reason: str | None = None,
    quantity_iri: str | None = None,
) -> dict[str, Any]:
    """Build the stable representation used for every scalar numeric metric."""

    if value is None and status == "ok":
        status = "not_computable"
        reason = reason or "value_unavailable"
    if value is not None:
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = True
        if not finite:
            value = None
            if status == "ok":
                status = "not_computable"
            reason = reason or "non_finite_value"
    payload: dict[str, Any] = {
        "value": value,
        "unit": unit,
        "status": status,
        "reason": reason,
    }
    if quantity_iri is not None:
        payload["quantity_iri"] = quantity_iri
    return payload


def make_record(
    *,
    record_id: str,
    record_type: str,
    cell_id: str,
    cycle_scope: Any,
    source_intervals: list[dict[str, Any]],
    attributes: dict[str, Any] | None = None,
    metrics: dict[str, dict[str, Any]] | None = None,
    series: dict[str, Any] | None = None,
    provider: str,
    method_name: str,
    provider_version: str,
    parameters: dict[str, Any] | None = None,
    references: list[str] | None = None,
    applicability_status: str = "applicable",
    applicability_reasons: list[str] | None = None,
    quality_status: str = "ok",
    quality_flags: list[str] | None = None,
    interpretation_limits: list[str] | None = None,
) -> dict[str, Any]:
    """Build one schema-complete analysis record."""

    return {
        "record_id": record_id,
        "record_type": record_type,
        "cell_id": cell_id,
        "cycle_scope": cycle_scope,
        "source_intervals": source_intervals,
        "attributes": attributes or {},
        "metrics": metrics or {},
        "series": series or {},
        "method": {
            "provider": provider,
            "name": method_name,
            "provider_version": provider_version,
            "parameters": parameters or {},
            "references": references or [],
        },
        "applicability": {
            "status": applicability_status,
            "reasons": applicability_reasons or [],
        },
        "quality": {"status": quality_status, "flags": quality_flags or []},
        "interpretation_limits": interpretation_limits or [],
    }


METRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["value", "unit", "status", "reason"],
    "properties": {
        "value": {"type": ["number", "integer", "null"]},
        "unit": {"type": "string"},
        "status": {"enum": ["ok", "not_computable"]},
        "reason": {"type": ["string", "null"]},
        "quantity_iri": {"type": "string"},
    },
    "additionalProperties": False,
    "allOf": [
        {
            "if": {"properties": {"status": {"const": "ok"}}},
            "then": {"properties": {"value": {"type": ["number", "integer"]}}},
        },
        {
            "if": {"properties": {"value": {"type": "null"}}},
            "then": {"properties": {"status": {"const": "not_computable"}}},
        },
    ],
}


SOURCE_INTERVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["start_row", "end_row", "start_time_s", "end_time_s"],
    "properties": {
        "role": {"type": "string"},
        "start_row": {"type": "integer"},
        "end_row": {"type": "integer"},
        "start_record": {"type": "integer"},
        "end_record": {"type": "integer"},
        "start_time_s": {"type": "number"},
        "end_time_s": {"type": "number"},
    },
    "additionalProperties": True,
}


RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "record_id",
        "record_type",
        "cell_id",
        "cycle_scope",
        "source_intervals",
        "attributes",
        "metrics",
        "series",
        "method",
        "applicability",
        "quality",
        "interpretation_limits",
    ],
    "properties": {
        "record_id": {"type": "string"},
        "record_type": {"type": "string"},
        "cell_id": {"type": "string"},
        "cycle_scope": {},
        "source_intervals": {"type": "array", "items": SOURCE_INTERVAL_SCHEMA},
        "attributes": {"type": "object"},
        "metrics": {"type": "object", "additionalProperties": METRIC_SCHEMA},
        "series": {"type": "object"},
        "method": {
            "type": "object",
            "required": ["provider", "name", "provider_version", "parameters", "references"],
            "properties": {
                "provider": {"type": "string"},
                "name": {"type": "string"},
                "provider_version": {"type": "string"},
                "parameters": {"type": "object"},
                "references": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "applicability": {
            "type": "object",
            "required": ["status", "reasons"],
            "properties": {
                "status": {
                    "enum": [
                        "applicable",
                        "not_computable",
                        "matched",
                        "unmatched",
                        "partial",
                    ]
                },
                "reasons": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "quality": {
            "type": "object",
            "required": ["status", "flags"],
            "properties": {
                "status": {"enum": ["ok", "warning"]},
                "flags": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
        "interpretation_limits": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


RESULTS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "run_id", "cell_id", "source", "configuration", "records"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "run_id": {"type": "string"},
        "cell_id": {"type": "string"},
        "source": {
            "type": "object",
            "required": [
                "input_path",
                "input_sha256",
                "normalized_data_file",
                "input_adapter",
                "input_provider",
                "input_report_file",
                "handoff_conformance_status",
                "metadata_file",
                "cycle_id_source",
            ],
            "properties": {
                "input_path": {"type": "string"},
                "input_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "normalized_data_file": {"const": "normalized_data.bdf.parquet"},
                "input_adapter": {"enum": ["bds", "bdf"]},
                "input_provider": {"type": "string", "minLength": 1},
                "input_report_file": {
                    "enum": ["bds_conversion_report.json", "bdf_validation_report.json"]
                },
                "handoff_conformance_status": {"type": "string", "minLength": 1},
                "metadata_file": {"const": "analysis_metadata.json"},
                "cycle_id_source": {"enum": ["source", "joined", "absent", "inferred", "unknown"]},
            },
            "additionalProperties": False,
        },
        "configuration": {"type": "object"},
        "records": {"type": "array", "items": RECORD_SCHEMA},
    },
    "additionalProperties": False,
}


COMPACT_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "record_id",
        "record_type",
        "cycle_scope",
        "attributes",
        "metrics",
        "series",
        "applicability",
        "quality",
        "interpretation_limits",
        "evidence",
    ],
    "properties": {
        "record_id": {"type": "string"},
        "record_type": {"type": "string"},
        "cycle_scope": {},
        "attributes": {"type": "object"},
        "metrics": {"type": "object", "additionalProperties": METRIC_SCHEMA},
        "series": {
            "type": "object",
            "required": ["available", "keys", "point_counts"],
            "properties": {
                "available": {"type": "boolean"},
                "keys": {"type": "array", "items": {"type": "string"}},
                "point_counts": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 0},
                },
            },
            "additionalProperties": False,
        },
        "applicability": RECORD_SCHEMA["properties"]["applicability"],
        "quality": RECORD_SCHEMA["properties"]["quality"],
        "interpretation_limits": {"type": "array", "items": {"type": "string"}},
        "evidence": {
            "type": "object",
            "required": ["file", "record_id", "source_interval_count"],
            "properties": {
                "file": {"type": "string"},
                "record_id": {"type": "string"},
                "source_interval_count": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


SUMMARY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "cell_id",
        "metadata_file",
        "evidence_file",
        "validation_file",
        "dimensions",
        "retrieval",
    ],
    "properties": {
        "schema_version": {"const": SUMMARY_SCHEMA_VERSION},
        "run_id": {"type": "string"},
        "cell_id": {"type": "string"},
        "metadata_file": {"type": "string"},
        "evidence_file": {"type": "string"},
        "validation_file": {"type": "string"},
        "dimensions": {
            "type": "object",
            "required": ["operation", "response", "evolution"],
            "properties": {
                "operation": {"type": "array", "items": COMPACT_RECORD_SCHEMA},
                "response": {"type": "array", "items": COMPACT_RECORD_SCHEMA},
                "evolution": {"type": "array", "items": COMPACT_RECORD_SCHEMA},
            },
            "additionalProperties": False,
        },
        "retrieval": {
            "type": "object",
            "required": ["join_key", "instructions", "evidence_record_count", "record_type_counts"],
            "properties": {
                "join_key": {"const": "record_id"},
                "instructions": {"type": "string"},
                "evidence_record_count": {"type": "integer", "minimum": 0},
                "record_type_counts": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 0},
                },
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


METADATA_FIELD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["value", "status", "source"],
    "properties": {
        "value": {},
        "status": {"enum": ["declared", "dataset_identifier", "unknown", "reported", "observed"]},
        "source": {"type": "string", "minLength": 1},
        "unit": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}


CHANNEL_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "quantity",
        "analysis_column",
        "source_column",
        "unit",
        "status",
        "origin",
        "finite_sample_fraction",
        "observed_min",
        "observed_max",
        "observed_increment_q10",
        "distinct_finite_values",
        "quality_flags",
    ],
    "properties": {
        "quantity": {"enum": ["time", "voltage", "current", "temperature"]},
        "analysis_column": {"type": ["string", "null"]},
        "source_column": {"type": ["string", "null"]},
        "unit": {"type": "string", "minLength": 1},
        "status": {"enum": ["available", "unavailable"]},
        "origin": {"type": "string", "minLength": 1},
        "finite_sample_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "observed_min": {"type": ["number", "null"]},
        "observed_max": {"type": ["number", "null"]},
        "observed_increment_q10": {"type": ["number", "null"], "minimum": 0},
        "distinct_finite_values": {"type": "integer", "minimum": 0},
        "quality_flags": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


CAPABILITY_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["column", "present", "usable", "finite_sample_fraction"],
    "properties": {
        "column": {"type": ["string", "null"]},
        "present": {"type": "boolean"},
        "usable": {"type": "boolean"},
        "finite_sample_fraction": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    },
    "additionalProperties": False,
}


CAPABILITIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["time", "current", "voltage", "temperature", "cycle", "step"],
    "properties": {
        name: CAPABILITY_ITEM_SCHEMA
        for name in ("time", "current", "voltage", "temperature", "cycle", "step")
    },
    "additionalProperties": False,
}


METADATA_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "cell",
        "test",
        "dataset",
        "channels",
        "semantics",
        "tool_sources",
        "limitations",
    ],
    "properties": {
        "schema_version": {"const": METADATA_SCHEMA_VERSION},
        "run_id": {"type": "string", "minLength": 1},
        "cell": {
            "type": "object",
            "required": [
                "cell_id",
                "chemistry",
                "manufacturer",
                "design",
                "form_factor",
                "nominal_capacity",
            ],
            "properties": {
                name: METADATA_FIELD_SCHEMA
                for name in (
                    "cell_id",
                    "chemistry",
                    "manufacturer",
                    "design",
                    "form_factor",
                    "nominal_capacity",
                )
            },
            "additionalProperties": False,
        },
        "test": {
            "type": "object",
            "required": [
                "declared_protocol_name",
                "cycler",
                "cycler_detection_confidence",
                "sheet_name",
                "start_datetime",
                "end_datetime",
                "observed_duration",
                "set_temperature",
            ],
            "properties": {
                name: METADATA_FIELD_SCHEMA
                for name in (
                    "declared_protocol_name",
                    "cycler",
                    "cycler_detection_confidence",
                    "sheet_name",
                    "start_datetime",
                    "end_datetime",
                    "observed_duration",
                    "set_temperature",
                )
            },
            "additionalProperties": False,
        },
        "dataset": {
            "type": "object",
            "required": [
                "name",
                "source_file",
                "source_sha256",
                "source_format",
                "row_count",
                "bds_schema_version",
                "bds_adapter_version",
                "bds_support_tier",
                "bds_evidence_tier",
                "input_adapter",
                "provider",
                "provider_version",
                "report_file",
                "conformance_status",
                "adapter_metadata",
                "capabilities",
                "source_artifacts",
            ],
            "properties": {
                "name": {"type": "string"},
                "source_file": {"type": "string"},
                "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "source_format": {"type": "string"},
                "row_count": {"type": "integer", "minimum": 0},
                "bds_schema_version": {"type": ["string", "null"]},
                "bds_adapter_version": {"type": ["string", "null"]},
                "bds_support_tier": {"type": ["string", "null"]},
                "bds_evidence_tier": {"type": ["string", "null"]},
                "input_adapter": {"enum": ["bds", "bdf"]},
                "provider": {"type": "string", "minLength": 1},
                "provider_version": {"type": ["string", "null"]},
                "report_file": {"type": "string", "minLength": 1},
                "conformance_status": {"type": ["string", "null"]},
                "adapter_metadata": {"type": "object"},
                "capabilities": CAPABILITIES_SCHEMA,
                "source_artifacts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["filename", "sha256", "role"],
                        "properties": {
                            "filename": {"type": "string", "minLength": 1},
                            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                            "role": {"const": "bdf_metadata_sidecar"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        "channels": {
            "type": "array",
            "minItems": 3,
            "items": CHANNEL_METADATA_SCHEMA,
        },
        "semantics": {
            "type": "object",
            "required": [
                "current_sign",
                "raw_current_sign",
                "current_sign_sanity",
                "cycle_id_source",
                "step_cycle_semantics",
                "ontology_context",
                "ontology_use",
            ],
            "properties": {
                "current_sign": {"const": "charge-positive"},
                "raw_current_sign": {"type": ["string", "null"]},
                "current_sign_sanity": {"type": "object"},
                "cycle_id_source": {"enum": ["source", "joined", "absent", "inferred", "unknown"]},
                "step_cycle_semantics": {"type": "object"},
                "ontology_context": {"type": "string"},
                "ontology_use": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "tool_sources": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["provider", "role"],
                "properties": {
                    "provider": {"type": "string"},
                    "role": {"type": "string"},
                    "report_file": {"type": "string"},
                    "metadata_status": {"type": "string"},
                    "execution_status": {"type": "string"},
                    "reference": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


VALIDATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "status",
        "artifacts",
        "software_versions",
        "provider_calls",
        "record_counts",
        "channel_capabilities",
        "quality_warnings",
        "schema_validation",
        "recomputation",
    ],
    "properties": {
        "schema_version": {"const": VALIDATION_SCHEMA_VERSION},
        "run_id": {"type": "string", "minLength": 1},
        "status": {"enum": ["ok", "warning"]},
        "artifacts": {
            "type": "object",
            "required": [
                "normalized_data.bdf.parquet",
                "analysis_metadata.json",
                "analysis_results.json",
                "analysis_evidence.json",
                "analysis_validation.json",
            ],
            "oneOf": [
                {"required": ["bds_conversion_report.json"]},
                {"required": ["bdf_validation_report.json"]},
            ],
            "minProperties": 6,
            "maxProperties": 6,
            "propertyNames": {
                "enum": [
                    "normalized_data.bdf.parquet",
                    "bds_conversion_report.json",
                    "bdf_validation_report.json",
                    "analysis_metadata.json",
                    "analysis_results.json",
                    "analysis_evidence.json",
                    "analysis_validation.json",
                ]
            },
            "additionalProperties": {
                "type": "object",
                "required": ["filename", "sha256"],
                "properties": {
                    "filename": {"type": "string"},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "size_bytes": {"type": "integer", "minimum": 0},
                    "digest_type": {"type": "string"},
                    "sha256_scope": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "software_versions": {
            "type": "object",
            "minProperties": 7,
            "additionalProperties": {"type": "string"},
        },
        "provider_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["provider", "method", "status"],
                "properties": {
                    "provider": {"type": "string"},
                    "method": {"type": "string"},
                    "status": {"enum": ["ok", "error", "not_invoked"]},
                },
                "additionalProperties": True,
            },
        },
        "record_counts": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "missing_inputs": {"type": "array", "items": {"type": "string"}},
        "channel_capabilities": CAPABILITIES_SCHEMA,
        "quality_warnings": {"type": "array", "items": {"type": "string"}},
        "schema_validation": {
            "type": "object",
            "required": [
                "draft",
                "analysis_results",
                "analysis_metadata",
                "analysis_evidence",
                "analysis_validation",
            ],
            "properties": {
                "draft": {"const": "2020-12"},
                **{
                    name: {
                        "type": "object",
                        "required": ["valid", "errors"],
                        "properties": {
                            "valid": {"type": "boolean"},
                            "errors": {"type": "array", "items": {"type": "string"}},
                        },
                        "additionalProperties": False,
                    }
                    for name in (
                        "analysis_results",
                        "analysis_metadata",
                        "analysis_evidence",
                        "analysis_validation",
                    )
                },
            },
            "additionalProperties": False,
        },
        "recomputation": {"type": "object", "minProperties": 1},
    },
    "additionalProperties": False,
}
