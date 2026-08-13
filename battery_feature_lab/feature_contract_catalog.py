"""Complete contract catalogue for retained deterministic feature columns.

The evidence layer uses the explicit whitelist in ``feature_contracts``. This module
adds non-AI-visible contracts for all other retained feature-table columns so the
full deterministic feature surface is documented without broadening what an LLM
may see.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from battery_feature_lab.feature_contracts import FEATURE_CONTRACTS, FeatureContract, contract_to_dict

_STRUCTURAL_COLUMNS = {
    "cell_id", "cycle_index", "reference_cycle", "target_cycle", "step_index",
    "step_type", "segment_index", "protocol", "protocol_family", "protocol_confidence",
}

_TABLE_TEMPLATES: dict[str, dict[str, Any]] = {
    "cycle_features": {
        "inputs": ("time_s", "voltage_v", "current_a", "temperature_c", "charge_capacity_ah", "discharge_capacity_ah", "step_type"),
        "method": "CycleSummaryFeaturizer deterministic cycle aggregation.",
        "parameters": ("nominal_capacity_ah", "min_points_per_cycle"),
        "applicability": "Requires a cycle with sufficient normalised samples; C-rate features additionally require nominal capacity.",
        "quality": "Cycle completeness and charge/discharge balance should be checked before cross-cycle interpretation.",
        "source_interval": "Contributing cycle interval.",
    },
    "delta_q_features": {
        "inputs": ("voltage_v", "discharge_capacity_ah", "cycle_index"),
        "method": "DeltaQFeaturizer common-voltage-grid comparison between configured discharge cycles.",
        "parameters": ("early_reference_cycle", "early_target_cycle", "delta_q_voltage_points", "min_points_for_curve"),
        "applicability": "Requires sufficient discharge-curve points and overlapping voltage domain in reference and target cycles.",
        "quality": "Rate, temperature and protocol comparability should be reviewed; parameter settings are part of provenance.",
        "source_interval": "Reference and target discharge-cycle intervals.",
    },
    "ica_dva_features": {
        "inputs": ("voltage_v", "charge_capacity_ah", "discharge_capacity_ah", "step_type", "cycle_index"),
        "method": "ICADVAFeaturizer resampling, smoothing, numerical differentiation and peak/summary extraction.",
        "parameters": ("voltage_grid_points", "capacity_grid_points", "smoothing_window", "smoothing_polyorder", "max_peaks", "min_points_for_curve", "peak_prominence_noise_multiplier"),
        "applicability": "Requires a sufficiently sampled charge or discharge segment with usable Q-V support.",
        "quality": "Derivative and peak descriptors are sampling- and smoothing-sensitive; untracked peak identity must not be assumed across cycles.",
        "source_interval": "Contributing charge or discharge segment.",
    },
    "relaxation_features": {
        "inputs": ("time_s", "voltage_v", "cycle_index", "step_index", "step_type"),
        "method": "RelaxationFeaturizer rest-segment summary, interpolation, slope and empirical single-exponential fit.",
        "parameters": (),
        "applicability": "Requires a rest segment with sufficient finite time-voltage samples.",
        "quality": "Cross-cycle comparison requires comparable pre-rest state, rest position and duration; exponential fit is empirical.",
        "source_interval": "Contributing rest-step interval.",
    },
    "stress_features": {
        "inputs": ("time_s", "current_a", "voltage_v", "temperature_c", "soc", "step_type"),
        "method": "StressHistogramFeaturizer time-history aggregation and distribution descriptors.",
        "parameters": ("nominal_capacity_ah", "histogram_bins", "high_soc_level"),
        "applicability": "Feature-specific inputs must be present; SOC-dependent features require explicitly provided SOC.",
        "quality": "Sampling density and observation coverage affect exposure summaries; missing inputs remain not computable.",
        "source_interval": "Full cell observation interval.",
    },
    "eis_features": {
        "inputs": ("frequency_hz", "z_real_ohm", "z_imag_ohm"),
        "method": "Descriptive EIS curve statistics only; no DRT inversion or equivalent-circuit fit.",
        "parameters": (),
        "applicability": "Requires finite frequency and complex-impedance columns.",
        "quality": "Descriptors are measurement-grid dependent and must not be interpreted as fitted physical parameters.",
        "source_interval": "Contributing EIS measurement group.",
    },
}


def write_complete_feature_contracts_json(path: str | Path, *, tables: dict[str, Any]) -> Path:
    """Write one contract for every retained deterministic feature column."""

    contracts: dict[tuple[str, str], FeatureContract] = dict(FEATURE_CONTRACTS)
    coverage: dict[str, dict[str, int]] = {}
    for source_table, frame in tables.items():
        template = _TABLE_TEMPLATES.get(source_table)
        if template is None or frame is None:
            continue
        feature_columns = [
            str(column)
            for column in getattr(frame, "columns", [])
            if str(column) not in _STRUCTURAL_COLUMNS
        ]
        for feature_name in feature_columns:
            key = (source_table, feature_name)
            if key not in contracts:
                contracts[key] = _generated_contract(source_table, feature_name, template)
        coverage[source_table] = {
            "feature_columns": len(feature_columns),
            "contracted_feature_columns": sum(
                1 for feature_name in feature_columns if (source_table, feature_name) in contracts
            ),
        }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.1",
        "policy": "Every retained deterministic feature column receives a contract. Only explicit contracts with ai_visible=true may enter AI-visible evidence.",
        "coverage": coverage,
        "contracts": [
            contract_to_dict(contract)
            for _, contract in sorted(contracts.items(), key=lambda item: item[0])
        ],
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _generated_contract(source_table: str, feature_name: str, template: dict[str, Any]) -> FeatureContract:
    return FeatureContract(
        contract_id=f"bfl:{source_table}:{feature_name}:v1.0.0",
        source_table=source_table,
        feature_name=feature_name,
        definition=_definition(source_table, feature_name),
        unit=_infer_unit(feature_name),
        inputs=tuple(template["inputs"]),
        method=str(template["method"]),
        method_version="1.0.0",
        parameter_names=tuple(template["parameters"]),
        applicability=str(template["applicability"]),
        quality=str(template["quality"]),
        source_interval=str(template["source_interval"]),
        interpretation_level="derived",
        ai_visible=False,
    )


def _definition(source_table: str, feature_name: str) -> str:
    name = feature_name.lower()
    if name.endswith("_count") or name in {"n_points", "eis_point_count"}:
        return f"Count of observations represented by `{feature_name}` in `{source_table}`."
    if "_hist_bin_" in name and name.endswith("_fraction"):
        return f"Fraction of finite observations falling inside the histogram bin represented by `{feature_name}`."
    if "_hist_bin_" in name and name.endswith(("_low", "_high")):
        return f"Histogram bin edge represented by `{feature_name}`."
    for statistic in ("mean", "std", "min", "max", "median", "q05", "q25", "q75", "q95", "skew", "kurtosis"):
        if name.endswith(f"_{statistic}"):
            return f"{statistic} summary statistic represented by `{feature_name}` over the contributing source interval."
    if "variance" in name:
        return f"Variance descriptor represented by `{feature_name}` for the producing feature family."
    if "duration" in name or "time_span" in name:
        return f"Elapsed-time descriptor represented by `{feature_name}` over the contributing interval."
    if "peak_" in name:
        return f"Peak descriptor `{feature_name}` extracted by the producing ICA/DVA method; peak rank is within-cycle and is not a cross-cycle identity."
    return f"Deterministic `{feature_name}` output produced in the `{source_table}` feature table."


def _infer_unit(feature_name: str) -> str | None:
    name = feature_name.lower()

    # Counts, standardised moments and source labels are dimensionless/text metadata.
    if name in {"n_points", "eis_point_count"} or name.endswith("_count"):
        return "count"
    if name.endswith(("_skew", "_kurtosis")):
        return "dimensionless"
    if name == "soc_source":
        return "category"
    if name.endswith("_fraction") or "efficiency" in name or name == "high_soc_level_used":
        return "fraction"
    if "low_frequency_current_power" in name:
        return "fraction"
    if name.endswith("_efc") or name == "equivalent_full_cycles":
        return "equivalent_full_cycles"

    # Histogram bin edges inherit the quantity represented by the prefix.
    if "_hist_bin_" in name and name.endswith(("_low", "_high")):
        prefix = name.split("_hist_bin_", 1)[0]
        return _base_quantity_unit(prefix)

    # Delta-Q descriptors.
    if name in {"delta_q_area_ah_v", "delta_q_abs_area_ah_v"}:
        return "Ah*V"
    if "delta_q" in name and "variance" in name:
        return "Ah^2" if "log_variance" not in name else "log10(Ah^2)"
    if name.startswith("delta_q_"):
        if "voltage" in name:
            return "V"
        return "Ah"

    # ICA/DVA curve and peak descriptors.
    if name == "ica_area":
        return "Ah"
    if name == "dva_area":
        return "V"
    if name.startswith("ica_dqdv"):
        return "Ah/V"
    if name.startswith("dva_dvdq"):
        return "V/Ah"
    if name.startswith("ica_peak_"):
        if name.endswith("_x") or name.endswith("_width_x"):
            return "V"
        if name.endswith(("_height", "_prominence")):
            return "Ah/V"
    if name.startswith("dva_peak_"):
        if name.endswith("_x") or name.endswith("_width_x"):
            return "Ah"
        if name.endswith(("_height", "_prominence")):
            return "V/Ah"

    # Relaxation descriptors.
    if "slope_v_per_s" in name:
        return "V/s"
    if name.endswith("_tau_s") or name.endswith("duration_s") or name.endswith("_time_span_s"):
        return "s"
    if name.startswith("rest_voltage") or name.endswith("_rmse_v") or name.endswith("_amplitude_v") or name.endswith("_v_inf_v"):
        return "V"

    # EIS descriptors.
    if "frequency" in name or name.endswith("_hz"):
        return "Hz"
    if "ohm" in name or name.startswith("eis_z_"):
        return "ohm"
    if "nyquist_slope" in name:
        return "dimensionless"

    # General scalar/summary families.
    if "temperature_integral_c_s" in name:
        return "degC*s"
    if "temperature" in name:
        return "degC"
    if name.endswith("_wh") or "energy_wh" in name:
        return "Wh"
    if name.endswith("_ah") or "capacity_ah" in name or name == "throughput_ah":
        return "Ah"
    if name.endswith("_a2") or "current_variance_a2" in name:
        return "A^2"
    if "c_rate_variance" in name:
        return "C^2"
    if "c_rate" in name:
        return "C"
    if name.endswith("_a") or "current_a" in name:
        return "A"
    if name.endswith("_v") or "voltage_v" in name:
        return "V"
    if name.endswith("_s"):
        return "s"
    if name.endswith("_h"):
        return "h"
    if name.startswith("soc_") or name == "soc":
        return "fraction"
    return "dimensionless"


def _base_quantity_unit(prefix: str) -> str:
    if prefix == "soc":
        return "fraction"
    if prefix == "voltage_v":
        return "V"
    if prefix == "temperature_c":
        return "degC"
    if prefix == "c_rate":
        return "C"
    if prefix in {"current_a", "abs_current_a"}:
        return "A"
    return "dimensionless"
