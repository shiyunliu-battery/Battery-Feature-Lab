"""Configuration and modality-specific schemas for EIS evidence."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from battery_feature_lab.analysis.schema import RECORD_SCHEMA, SUMMARY_SCHEMA

CONDITION_UNITS = {
    "temperature_deg_c": "degC",
    "soc_fraction": "1",
    "rest_time_s": "s",
    "ac_amplitude_v": "V",
    "ac_amplitude_a": "A",
    "dc_voltage_v": "V",
    "cycle_index": "1",
    "sequence_index": "1",
    "elapsed_time_s": "s",
    "cell_id": None,
    "instrument": None,
    "protocol_id": None,
}


@dataclass(frozen=True)
class EISConfig:
    output_dir: Path = Path("bfl_eis_outputs")
    sheet: str | int | None = None
    single_spectrum: bool = False
    group_columns: tuple[str, ...] = ()
    columns: dict[str, str] = field(default_factory=dict)
    phase_unit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_artifact: dict[str, str] = field(default_factory=dict)
    probe_frequencies_hz: tuple[float, ...] = (1.0, 10.0, 1000.0)
    run_kk: bool = True
    run_fit: bool = True
    fit_options: dict[str, Any] = field(default_factory=dict)
    kk_options: dict[str, Any] = field(default_factory=dict)
    minimum_provider_points: int = 12
    kk_residual_limit: float = 0.05
    fit_nrmse_limit: float = 0.05
    comparison_pairs: tuple[tuple[str, str], ...] = ()
    comparison_kind: str = "repeat"
    comparison_tolerances: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.phase_unit not in {None, "deg", "rad"}:
            raise ValueError("phase_unit must be deg, rad or None")
        if set(self.columns) - {"frequency", "real", "imaginary", "magnitude", "phase"}:
            raise ValueError("Unknown EIS column mapping key")
        if self.comparison_kind not in {"repeat", "ageing", "soc", "temperature"}:
            raise ValueError("Unknown comparison_kind")
        if isinstance(self.minimum_provider_points, bool) or self.minimum_provider_points < 5:
            raise ValueError("minimum_provider_points must be an integer of at least five")
        if int(self.minimum_provider_points) != self.minimum_provider_points:
            raise ValueError("minimum_provider_points must be an integer")
        numbers = [*self.probe_frequencies_hz, self.kk_residual_limit, self.fit_nrmse_limit]
        if any(not math.isfinite(float(x)) or float(x) <= 0 for x in numbers):
            raise ValueError("Probe frequencies and residual limits must be finite and positive")
        labels = [format(float(x), ".15g") for x in self.probe_frequencies_hz]
        if len(labels) != len(set(labels)):
            raise ValueError("Probe frequencies must have distinct metric labels")
        if set(self.comparison_tolerances) - set(CONDITION_UNITS):
            raise ValueError("Unknown comparison tolerance")
        if any(CONDITION_UNITS[name] is None for name in self.comparison_tolerances):
            raise ValueError("Comparison tolerances apply only to numeric conditions")
        if any(not math.isfinite(float(x)) or x < 0 for x in self.comparison_tolerances.values()):
            raise ValueError("Comparison tolerances must be finite and non-negative")
        if set(self.kk_options) - {"c", "max_M", "fit_type", "add_cap"}:
            raise ValueError("Unknown Lin-KK option")
        if self.kk_options.get("fit_type", "complex") not in {"real", "imag", "complex"}:
            raise ValueError("Unknown Lin-KK fit_type")
        maximum = self.kk_options.get("max_M", 50)
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 3:
            raise ValueError("Lin-KK max_M must be an integer of at least three")
        cutoff = self.kk_options.get("c", 0.85)
        if cutoff is not None and not (math.isfinite(float(cutoff)) and 0 < cutoff < 1):
            raise ValueError("Lin-KK c must be between zero and one, or None")
        if set(self.fit_options) - {
            "tail",
            "n_starts",
            "seed",
            "max_nfev",
            "weight_func",
            "weight_eps",
            "loss",
            "f_scale",
            "use_adaptive_bounds",
            "lb",
            "ub",
        }:
            raise ValueError("Unknown ECM fit option")
        if self.fit_options.get("tail", "cpe") not in {"cpe", "warburg"}:
            raise ValueError("Unknown ECM tail")
        for key in ("n_starts", "max_nfev"):
            value = self.fit_options.get(key, 1)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"ECM {key} must be a positive integer")
        if not isinstance(self.metadata.get("defaults", {}), dict) or not isinstance(
            self.metadata.get("spectra", {}), dict
        ):
            raise TypeError("EIS metadata defaults and spectra must be objects")
        if set(self.metadata) - {"defaults", "spectra"}:
            raise ValueError("EIS metadata must contain only defaults and spectra")
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in self.comparison_pairs):
            raise ValueError("Each comparison must name two distinct spectra")
        json.dumps(self.configuration(), allow_nan=False)

    def configuration(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "output_dir"}


EIS_RECORD_SCHEMA = copy.deepcopy(RECORD_SCHEMA)
EIS_RECORD_SCHEMA["properties"]["source_intervals"]["items"] = {
    "type": "object",
    "required": ["source_file", "source_sha256", "sheet", "source_rows", "row_index_basis"],
    "properties": {
        "source_file": {"type": "string"},
        "source_sha256": {"type": "string"},
        "sheet": {"type": ["string", "integer", "null"]},
        "source_rows": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "row_index_basis": {"const": "zero_based_BDS_parsed_table"},
    },
    "additionalProperties": False,
}

EIS_EVIDENCE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "modality", "run_id", "configuration", "records"],
    "properties": {
        "schema_version": {"const": "bfl.eis-analysis/0.1"},
        "modality": {"const": "eis"},
        "run_id": {"type": "string"},
        "configuration": {"type": "object"},
        "records": {"type": "array", "items": EIS_RECORD_SCHEMA},
    },
    "additionalProperties": False,
}

EIS_SUMMARY_SCHEMA = copy.deepcopy(SUMMARY_SCHEMA)
EIS_SUMMARY_SCHEMA["properties"]["schema_version"] = {"const": "bfl.eis-summary/0.1"}

EIS_METADATA_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "source", "spectra"],
    "properties": {
        "schema_version": {"const": "bfl.eis-metadata/0.1"},
        "source": {"type": "object"},
        "spectra": {"type": "array", "items": {"type": "object"}},
    },
    "additionalProperties": False,
}

EIS_VALIDATION_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version",
        "run_id",
        "status",
        "artifacts",
        "schema_validation",
        "row_accounting",
        "provider_calls",
    ],
    "properties": {
        "schema_version": {"const": "bfl.eis-validation/0.1"},
        "run_id": {"type": "string"},
        "status": {"enum": ["ok", "warning"]},
        "artifacts": {"type": "object"},
        "schema_validation": {"type": "object"},
        "row_accounting": {"type": "object"},
        "provider_calls": {"type": "array"},
        "software_versions": {"type": "object"},
    },
    "additionalProperties": False,
}
