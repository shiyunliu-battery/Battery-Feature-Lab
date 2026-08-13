"""Controlled contracts for AI-visible Battery Feature Lab features.

A feature may enter the evidence layer only when it has an explicit contract here.
This is intentionally a whitelist: adding a numeric column to a feature table does
not make it AI-visible.

Contracts describe what a feature means, how it is computed, the inputs and
configuration that affect it, when it is applicable, the quality checks expected
before interpretation, the source interval used for provenance, and the permitted
interpretation level.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class FeatureContract:
    """Machine-readable contract for one feature exposed to the evidence layer."""

    contract_id: str
    source_table: str
    feature_name: str
    definition: str
    unit: str | None
    inputs: tuple[str, ...]
    method: str
    method_version: str
    parameter_names: tuple[str, ...]
    applicability: str
    quality: str
    source_interval: str
    interpretation_level: str
    ai_visible: bool = True


def _contract(
    source_table: str,
    feature_name: str,
    *,
    definition: str,
    unit: str | None,
    inputs: tuple[str, ...],
    method: str,
    parameter_names: tuple[str, ...] = (),
    applicability: str,
    quality: str,
    source_interval: str,
    interpretation_level: str = "derived",
    ai_visible: bool = True,
    method_version: str = "1.0.0",
) -> FeatureContract:
    return FeatureContract(
        contract_id=f"bfl:{source_table}:{feature_name}:v{method_version}",
        source_table=source_table,
        feature_name=feature_name,
        definition=definition,
        unit=unit,
        inputs=inputs,
        method=method,
        method_version=method_version,
        parameter_names=parameter_names,
        applicability=applicability,
        quality=quality,
        source_interval=source_interval,
        interpretation_level=interpretation_level,
        ai_visible=ai_visible,
    )


_CONTRACTS = [
    _contract(
        "cycle_features",
        "discharge_capacity_retention",
        definition="Ratio of the last usable discharge capacity to the first usable discharge capacity in the analysed interval.",
        unit="fraction",
        inputs=("discharge_capacity_ah", "cycle_index"),
        method="Endpoint ratio over ordered cycle-level discharge-capacity observations.",
        applicability="Requires at least two finite discharge-capacity observations from comparable cycling conditions.",
        quality="Endpoint cycles should be complete and capacity-balanced; protocol changes must be treated as a comparability warning.",
        source_interval="Full source interval spanning the first and last contributing cycles.",
        interpretation_level="derived",
    ),
    _contract(
        "cycle_features",
        "coulombic_efficiency",
        definition="Discharge capacity divided by charge capacity for one cycle.",
        unit="fraction",
        inputs=("charge_capacity_ah", "discharge_capacity_ah"),
        method="Cycle-level capacity ratio.",
        applicability="Requires positive finite charge and discharge capacity for the same cycle.",
        quality="Cycle should contain complete charge and discharge segments; values outside physically plausible bounds require review.",
        source_interval="Full contributing cycle.",
        interpretation_level="derived",
    ),
    _contract(
        "cycle_features",
        "cv_capacity_fraction",
        definition="Legacy near-maximum-voltage proxy for the fraction of charge capacity accumulated near the charge-voltage maximum.",
        unit="fraction",
        inputs=("voltage_v", "charge_capacity_ah"),
        method="Capacity within 10 mV of observed maximum charge voltage divided by total charge capacity.",
        applicability="Not a validated constant-voltage phase detector.",
        quality="Withheld from AI-visible evidence until protocol-segmented CV capacity is implemented.",
        source_interval="Charge portion of one cycle.",
        interpretation_level="derived",
        ai_visible=False,
    ),
    _contract(
        "delta_q_features",
        "delta_q_abs_area_ah_v",
        definition="Integral of the absolute difference between target- and reference-cycle discharge capacity curves on their overlapping voltage domain.",
        unit="Ah*V",
        inputs=("voltage_v", "discharge_capacity_ah", "cycle_index"),
        method="Interpolate Q(V) for configured reference and target discharge cycles onto a common voltage grid and integrate |ΔQ(V)|.",
        parameter_names=("early_reference_cycle", "early_target_cycle", "delta_q_voltage_points", "min_points_for_curve"),
        applicability="Reference and target discharge curves must have sufficient points and overlapping voltage range; rate, temperature and protocol comparability should be checked.",
        quality="Both cycles should be complete and comparable; large protocol or condition changes make the result conditional rather than diagnostic.",
        source_interval="Reference and target discharge-cycle intervals.",
        interpretation_level="derived",
    ),
    _contract(
        "delta_q_features",
        "delta_q_variance",
        definition="Sample variance of ΔQ(V) between configured target and reference discharge cycles.",
        unit="Ah^2",
        inputs=("voltage_v", "discharge_capacity_ah", "cycle_index"),
        method="Interpolate Q(V) on a common voltage grid, compute target minus reference, then sample variance.",
        parameter_names=("early_reference_cycle", "early_target_cycle", "delta_q_voltage_points", "min_points_for_curve"),
        applicability="Same comparability requirements as other ΔQ(V) features.",
        quality="Requires sufficient overlapping curve support and comparable test conditions.",
        source_interval="Reference and target discharge-cycle intervals.",
        interpretation_level="derived",
    ),
    _contract(
        "delta_q_features",
        "delta_q_l2",
        definition="Euclidean norm of ΔQ(V) samples on the configured common voltage grid.",
        unit="Ah",
        inputs=("voltage_v", "discharge_capacity_ah", "cycle_index"),
        method="Interpolate Q(V), compute ΔQ(V), then sqrt(sum(ΔQ²)).",
        parameter_names=("early_reference_cycle", "early_target_cycle", "delta_q_voltage_points", "min_points_for_curve"),
        applicability="Same comparability requirements as other ΔQ(V) features.",
        quality="Grid density affects the raw L2 magnitude; interpret only with recorded parameters.",
        source_interval="Reference and target discharge-cycle intervals.",
        interpretation_level="derived",
    ),
    _contract(
        "ica_dva_features",
        "ica_area_change",
        definition="Change in integrated incremental-capacity area between the first and last comparable segment of the same step type.",
        unit="Ah",
        inputs=("voltage_v", "charge_capacity_ah", "discharge_capacity_ah", "step_type", "cycle_index"),
        method="Resample Q(V), smooth, differentiate to dQ/dV, integrate, and subtract matched endpoint values within one step type.",
        parameter_names=("voltage_grid_points", "smoothing_window", "smoothing_polyorder", "min_points_for_curve"),
        applicability="Requires comparable charge-to-charge or discharge-to-discharge segments; protocol changes make the comparison conditional.",
        quality="Derivative features are sensitive to sampling and smoothing; recorded grid and smoothing parameters are part of the evidence.",
        source_interval="First and last contributing same-step-type cycle intervals.",
        interpretation_level="derived",
    ),
    _contract(
        "ica_dva_features",
        "dva_area_change",
        definition="Change in integrated absolute differential-voltage area between the first and last comparable segment of the same step type.",
        unit="V",
        inputs=("voltage_v", "charge_capacity_ah", "discharge_capacity_ah", "step_type", "cycle_index"),
        method="Resample V(Q), smooth, differentiate to dV/dQ, integrate absolute derivative, and subtract endpoint values within one step type.",
        parameter_names=("capacity_grid_points", "smoothing_window", "smoothing_polyorder", "min_points_for_curve"),
        applicability="Requires comparable same-step-type segments; protocol changes make the comparison conditional.",
        quality="Derivative features are sensitive to sampling and smoothing; do not infer a unique degradation mechanism from this feature alone.",
        source_interval="First and last contributing same-step-type cycle intervals.",
        interpretation_level="derived",
    ),
    _contract(
        "relaxation_features",
        "rest_exp_tau_s_change",
        definition="Change in empirical single-exponential relaxation time constant for the same step index between endpoint cycles.",
        unit="s",
        inputs=("time_s", "voltage_v", "cycle_index", "step_index"),
        method="Fit V(t)=V_inf+A*exp(-t/tau) to rest segments and subtract endpoint tau values matched by step index.",
        applicability="Requires the same rest position/step index to represent comparable pre-rest conditions; otherwise interpretation is conditional.",
        quality="Empirical one-exponential fit only; review fit RMSE and protocol consistency before interpretation.",
        source_interval="Matched rest-step intervals at the first and last contributing cycles.",
        interpretation_level="derived",
    ),
    _contract(
        "relaxation_features",
        "rest_voltage_delta_v_change",
        definition="Change in rest-segment voltage recovery/drop between endpoint cycles for the same step index.",
        unit="V",
        inputs=("time_s", "voltage_v", "cycle_index", "step_index"),
        method="Subtract initial rest voltage from final rest voltage per segment, then compare matched step-index endpoints.",
        applicability="Requires comparable rest duration and pre-rest state at the same step index.",
        quality="Protocol or rest-duration changes make the comparison conditional.",
        source_interval="Matched rest-step intervals at the first and last contributing cycles.",
        interpretation_level="derived",
    ),
    _contract(
        "relaxation_features",
        "rest_voltage_linear_slope_v_per_s_change",
        definition="Change in linear rest-voltage slope between endpoint cycles for the same step index.",
        unit="V/s",
        inputs=("time_s", "voltage_v", "cycle_index", "step_index"),
        method="Least-squares linear slope across each rest segment, compared across matched step-index endpoints.",
        applicability="Requires comparable rest segments and pre-rest state.",
        quality="Sensitive to rest duration and non-linear relaxation; treat as a descriptive descriptor.",
        source_interval="Matched rest-step intervals at the first and last contributing cycles.",
        interpretation_level="derived",
    ),
    _contract(
        "stress_features",
        "high_soc_rest_fraction",
        definition="Time-weighted fraction of observed time spent at rest while an explicitly provided SOC signal is at or above the configured high-SOC threshold.",
        unit="fraction",
        inputs=("time_s", "current_a", "soc", "step_type"),
        method="Time-weighted fraction over samples classified as rest and SOC >= configured threshold.",
        parameter_names=("high_soc_level",),
        applicability="Only applicable when SOC is explicitly supplied by the input data; BFL does not infer SOC from voltage or capacity.",
        quality="Missing explicit SOC yields not-computable rather than a surrogate estimate.",
        source_interval="Full cell observation interval.",
        interpretation_level="derived",
    ),
    _contract(
        "stress_features",
        "max_instant_discharge_c_rate",
        definition="Maximum observed discharge-current magnitude divided by nominal capacity.",
        unit="C",
        inputs=("current_a", "nominal_capacity_ah"),
        method="Maximum discharge |I| / nominal capacity.",
        parameter_names=("nominal_capacity_ah",),
        applicability="Requires a user-supplied nominal capacity.",
        quality="Instantaneous extrema are sensitive to spikes; compare with raw data and sampling rate when used for limits.",
        source_interval="Full cell observation interval.",
        interpretation_level="derived",
    ),
    _contract(
        "stress_features",
        "throughput_ah",
        definition="Time integral of absolute current over the observation interval.",
        unit="Ah",
        inputs=("time_s", "current_a"),
        method="Sum |I|*dt using elapsed sample intervals.",
        applicability="Requires a valid monotonically increasing time axis after BDS normalisation.",
        quality="Large gaps or sparse sampling should be reviewed because interval integration assumes the recorded current persists between samples.",
        source_interval="Full cell observation interval.",
        interpretation_level="derived",
    ),
    _contract(
        "stress_features",
        "temperature_c_mean",
        definition="Arithmetic mean of available cell-temperature observations.",
        unit="degC",
        inputs=("temperature_c",),
        method="Mean of finite temperature samples.",
        applicability="Requires a temperature channel.",
        quality="Sampling-weighted arithmetic mean; not a time-weighted thermal-dose metric.",
        source_interval="Full cell observation interval.",
        interpretation_level="derived",
    ),
    _contract(
        "degradation_tags",
        "capacity_decreasing_trend",
        definition="Statistically significant monotonic decreasing trend in normalised discharge capacity.",
        unit=None,
        inputs=("cycle_index", "discharge_capacity_ah"),
        method="Kendall trend test with Sen median pairwise slope.",
        parameter_names=("trend_p_value_alpha", "min_trend_points"),
        applicability="Requires at least the configured minimum number of finite cycle-capacity observations.",
        quality="This is an observed trend signal only; it does not identify LLI, LAM or another degradation mechanism.",
        source_interval="First-to-last cycle interval used by the trend test.",
        interpretation_level="observation",
    ),
    _contract(
        "degradation_tags",
        "high_soc_rest_exposure",
        definition="Configured or batch-relative threshold flag on high-SOC rest exposure computed from explicit SOC.",
        unit=None,
        inputs=("high_soc_rest_fraction",),
        method="Threshold comparison on the contracted high_soc_rest_fraction feature.",
        parameter_names=("high_soc_rest_fraction_threshold", "stress_percentile"),
        applicability="Only applicable when high_soc_rest_fraction is computable from explicit SOC.",
        quality="Stress-context flag, not a degradation-mechanism diagnosis.",
        source_interval="Full cell observation interval.",
        interpretation_level="observation",
    ),
    _contract(
        "degradation_tags",
        "dynamic_current_variance",
        definition="Configured or batch-relative outlier flag on C-rate variance.",
        unit=None,
        inputs=("c_rate_variance",),
        method="Absolute threshold or robust batch median-plus-MAD rule.",
        parameter_names=("c_rate_variance_threshold", "stress_mad_threshold"),
        applicability="Requires nominal capacity and finite C-rate variance.",
        quality="Usage-stress observation only; no automatic mechanism label is permitted.",
        source_interval="Full cell observation interval.",
        interpretation_level="observation",
    ),
    _contract(
        "degradation_tags",
        "high_instantaneous_discharge_rate",
        definition="Flag indicating observed discharge C-rate exceeded the configured fraction of a user-provided datasheet limit.",
        unit=None,
        inputs=("max_instant_discharge_c_rate", "datasheet_max_discharge_c_rate"),
        method="Observed maximum discharge C-rate compared with configured datasheet fraction.",
        parameter_names=("datasheet_max_discharge_c_rate", "max_discharge_c_rate_fraction"),
        applicability="Requires user-provided nominal capacity and datasheet maximum discharge C-rate.",
        quality="Specification-relative usage flag, not a degradation-mechanism diagnosis.",
        source_interval="Full cell observation interval.",
        interpretation_level="observation",
    ),
]

FEATURE_CONTRACTS: dict[tuple[str, str], FeatureContract] = {
    (contract.source_table, contract.feature_name): contract for contract in _CONTRACTS
}


def get_feature_contract(source_table: str, feature_name: str) -> FeatureContract | None:
    """Return the exact registered contract; there is deliberately no generic fallback."""

    return FEATURE_CONTRACTS.get((source_table, feature_name))


def resolve_method_parameters(
    contract: FeatureContract,
    configs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve only parameters explicitly named by the contract."""

    configs = configs or {}
    return {name: configs.get(name) for name in contract.parameter_names if name in configs}


def contract_to_dict(contract: FeatureContract) -> dict[str, Any]:
    payload = asdict(contract)
    payload["inputs"] = list(contract.inputs)
    payload["parameter_names"] = list(contract.parameter_names)
    return payload


def write_feature_contracts_json(path: str | Path) -> Path:
    """Write the complete registry so evidence records can reference stable contract IDs."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "policy": "Only features with ai_visible=true may enter AI-visible evidence.",
        "contracts": [contract_to_dict(contract) for contract in _CONTRACTS],
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
