"""Short-window operating and electrochemical response summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import polars as pl
import pyprobe

from battery_feature_lab.analysis.adapters import temperature_column
from battery_feature_lab.analysis.numerics import (
    json_number,
    previous_intervals,
    weighted_quantile,
)
from battery_feature_lab.analysis.schema import (
    ANALYSIS_POLICY_VERSION,
    AnalysisConfig,
    make_record,
    metric,
    seconds_label,
)

PYPROBE_REFERENCE = "https://doi.org/10.21105/joss.07474"
def analyze_directional_energy(
    frame: pl.DataFrame,
    *,
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
) -> dict[str, Any]:
    """Summarize directional charge, energy, voltage, and power without cycle claims."""

    selected = _select_cycle(frame, cycle_id)
    if selected.height < 2:
        return _unavailable_directional_record(cell_id, cycle_id, "fewer_than_two_samples")
    time = selected["test_time_s"].cast(pl.Float64).to_numpy()
    current = selected["current_a"].cast(pl.Float64).to_numpy()
    voltage = selected["voltage_v"].cast(pl.Float64).to_numpy()
    intervals = previous_intervals(
        time,
        current,
        sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
    )
    dt = intervals.durations_s
    held_current = current[:-1]
    held_voltage = voltage[:-1]
    positive_time = np.isfinite(dt) & (dt > 0)
    joint = positive_time & np.isfinite(held_current) & np.isfinite(held_voltage)
    charge = joint & (held_current > config.rest_current_threshold_a)
    discharge = joint & (held_current < -config.rest_current_threshold_a)
    active = charge | discharge

    charge_ah = _sum(held_current[charge] * dt[charge]) / 3600.0
    discharge_ah = _sum(-held_current[discharge] * dt[discharge]) / 3600.0
    charge_wh = _sum(held_voltage[charge] * held_current[charge] * dt[charge]) / 3600.0
    discharge_wh = _sum(-held_voltage[discharge] * held_current[discharge] * dt[discharge]) / 3600.0
    charge_voltage = charge_wh / charge_ah if charge_ah > 0 else None
    discharge_voltage = discharge_wh / discharge_ah if discharge_ah > 0 else None
    voltage_gap = (
        charge_voltage - discharge_voltage
        if charge_voltage is not None and discharge_voltage is not None
        else None
    )
    maximum_throughput = max(charge_ah, discharge_ah)
    balance = min(charge_ah, discharge_ah) / maximum_throughput if maximum_throughput > 0 else None
    balance_min = config.policy("directional_balance_min")
    balanced = balance is not None and balance >= balance_min
    energy_return = discharge_wh / charge_wh if balanced and charge_wh > 0 else None
    voltage_quantiles = weighted_quantile(held_voltage[joint], dt[joint], [0.05, 0.5, 0.95])
    power_quantile = weighted_quantile(
        np.abs(held_voltage[active] * held_current[active]),
        dt[active],
        [0.95],
    )[0]
    total_positive_duration = float(np.sum(dt[positive_time]))
    included_duration = float(np.sum(dt[joint]))
    excluded_duration = max(total_positive_duration - included_duration, 0.0)
    flags: list[str] = []
    if excluded_duration > 0:
        flags.append("excluded_non_finite_voltage_or_current_duration")
    if intervals.sampling_interval_outlier_count:
        flags.append("sampling_interval_outlier")
    if charge_ah <= 0 or discharge_ah <= 0:
        flags.append("charge_or_discharge_direction_missing")
    if balance is not None and not balanced:
        flags.append("directional_throughput_imbalance")

    rows = selected["_source_row"].to_numpy()
    source_records = (
        selected["record_index"].to_numpy() if "record_index" in selected.columns else rows
    )
    return make_record(
        record_id=f"response.directional_energy_summary:{cycle_id}",
        record_type="response.directional_energy_summary",
        cell_id=cell_id,
        cycle_scope=cycle_id,
        source_intervals=[
            {
                "start_row": int(rows[0]),
                "end_row": int(rows[-1]),
                "start_record": int(source_records[0]),
                "end_record": int(source_records[-1]),
                "start_time_s": float(time[0]),
                "end_time_s": float(time[-1]),
            }
        ],
        attributes={
            "balanced_window_threshold": balance_min,
            "current_sign": "charge-positive",
            "interval_semantics": "x[i] holds on [t[i], t[i+1])",
            "reference_frame": {
                "current_sign": "charge-positive",
                "scope": "selected_representative_cycle"
                if cycle_id is not None
                else "complete_observation_window",
                "state_axis": "observed_time",
            },
            "confidence": {
                "level": "medium" if flags else "high",
                "basis": ["finite_voltage_current_intervals", "previous_value_zoh", *flags],
                "not_a_probability": True,
            },
        },
        metrics={
            "charge_throughput": metric(charge_ah, "Ah"),
            "discharge_throughput": metric(discharge_ah, "Ah"),
            "net_capacity_change": metric(charge_ah - discharge_ah, "Ah"),
            "charge_energy": metric(charge_wh, "Wh"),
            "discharge_energy": metric(discharge_wh, "Wh"),
            "charge_mean_voltage": metric(charge_voltage, "V"),
            "discharge_mean_voltage": metric(discharge_voltage, "V"),
            "directional_mean_voltage_gap": metric(voltage_gap, "V"),
            "charge_discharge_throughput_balance": metric(balance, "1"),
            "balanced_window_energy_return_ratio": metric(
                energy_return,
                "1",
                status="ok" if energy_return is not None else "not_computable",
                reason=(
                    None
                    if energy_return is not None
                    else "charge and discharge throughput are not both present and balanced within 5%"
                ),
            ),
            "voltage_time_q05": metric(json_number(voltage_quantiles[0]), "V"),
            "voltage_time_q50": metric(json_number(voltage_quantiles[1]), "V"),
            "voltage_time_q95": metric(json_number(voltage_quantiles[2]), "V"),
            "absolute_power_time_q95": metric(json_number(power_quantile), "W"),
            "included_duration": metric(included_duration, "s"),
            "excluded_duration": metric(excluded_duration, "s"),
            "sampling_interval_outlier_count": metric(
                intervals.sampling_interval_outlier_count, "1"
            ),
            "sampling_interval_outlier_duration": metric(
                intervals.sampling_interval_outlier_duration_s, "s"
            ),
            "maximum_sampling_interval_outlier": metric(
                intervals.max_sampling_interval_outlier_s, "s"
            ),
        },
        provider="BFL",
        method_name="directional_previous_zoh_energy_voltage_v1",
        provider_version="0.4.0",
        parameters={
            "rest_current_threshold_a": config.rest_current_threshold_a,
            "sampling_interval_outlier_factor": (config.sampling_interval_outlier_factor),
            "sampling_interval_outlier_detection": (
                "isolated_interval_vs_nearest_positive_neighbors_v1"
            ),
            "sampling_interval_outliers_included_in_integrals": True,
            "analysis_policy_version": ANALYSIS_POLICY_VERSION,
            "balanced_window_minimum_throughput_ratio": balance_min,
        },
        references=[
            "https://doi.org/10.1038/s41597-024-03831-x",
            "https://rovi-org.github.io/battery-data-toolkit/user-guide/post-processing/index.html",
        ],
        applicability_status=(
            "applicable" if charge_ah > 0 or discharge_ah > 0 else "not_computable"
        ),
        applicability_reasons=([] if charge_ah > 0 or discharge_ah > 0 else ["no_active_current"]),
        quality_status="warning" if flags else "ok",
        quality_flags=flags,
        interpretation_limits=[
            "Directional totals describe the selected observation window, not a complete cycle.",
            "The directional mean-voltage gap combines polarization with any charge/discharge state-window mismatch.",
            "The balanced-window energy return ratio is descriptive and is not reported as cycle efficiency.",
        ],
    )


def analyze_relaxation_summary(
    rest_records: list[dict[str, Any]],
    *,
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
) -> dict[str, Any]:
    """Summarize load-conditioned fixed-time voltage recovery."""

    all_rests = [
        record for record in rest_records if record["record_type"] == "response.rest_and_thermal"
    ]
    applicable_rests = [
        record
        for record in all_rests
        if record["source_intervals"] and record["applicability"]["status"] == "applicable"
    ]
    rests = [
        record
        for record in applicable_rests
        if record["attributes"].get("previous_phase") in {"charge", "discharge"}
    ]
    rejection_counts = Counter(
        reason
        for record in all_rests
        if record["applicability"]["status"] != "applicable"
        for reason in record["applicability"]["reasons"]
    )
    metrics: dict[str, dict[str, Any]] = {
        "rest_segment_count": metric(len(applicable_rests), "1"),
        "phase_conditioned_rest_segment_count": metric(len(rests), "1"),
        "unconditioned_rest_segment_count": metric(len(applicable_rests) - len(rests), "1"),
        "rejected_rest_segment_count": metric(len(all_rests) - len(applicable_rests), "1"),
    }
    source_intervals: list[dict[str, Any]] = []
    contributing_counts: dict[str, dict[str, int]] = {}
    any_phase_conditioned_sufficient = False
    flags = sorted(
        {
            flag
            for record in rests
            for flag in record["quality"]["flags"]
            if not flag.startswith("temperature_")
        }
    )
    if rejection_counts:
        flags.append("rest_segments_rejected")
    minimum_count = int(config.policy("relaxation_summary_min_count"))
    for checkpoint in config.relaxation_checkpoints_s:
        checkpoint_label = seconds_label(checkpoint)
        name = f"voltage_change_at_{checkpoint_label}s"
        eligible = [
            record
            for record in rests
            if record["metrics"].get(name, {}).get("status") == "ok"
            and record["metrics"][name]["value"] is not None
        ]
        values = np.asarray([record["metrics"][name]["value"] for record in eligible], dtype=float)
        source_intervals.extend(
            interval for record in eligible for interval in record["source_intervals"]
        )
        metrics[f"eligible_rest_count_{checkpoint_label}s"] = metric(len(values), "1")
        metrics[f"absolute_voltage_change_q50_{checkpoint_label}s"] = _quantile_metric(
            np.abs(values), 0.5, minimum_count=minimum_count
        )
        metrics[f"absolute_voltage_change_q95_{checkpoint_label}s"] = _quantile_metric(
            np.abs(values), 0.95, minimum_count=minimum_count
        )

        recoveries: list[float] = []
        counts = {"after_charge": 0, "after_discharge": 0, "other": 0}
        for record in eligible:
            delta = float(record["metrics"][name]["value"])
            previous = record["attributes"].get("previous_phase")
            if previous == "charge":
                recoveries.append(-delta)
                counts["after_charge"] += 1
            elif previous == "discharge":
                recoveries.append(delta)
                counts["after_discharge"] += 1
            else:
                counts["other"] += 1
        recovery_array = np.asarray(recoveries, dtype=float)
        any_phase_conditioned_sufficient |= len(recovery_array) >= minimum_count
        metrics[f"polarization_recovery_q50_{checkpoint_label}s"] = _quantile_metric(
            recovery_array, 0.5, minimum_count=minimum_count
        )
        metrics[f"positive_recovery_fraction_{checkpoint_label}s"] = metric(
            (
                float(np.mean(recovery_array > 0))
                if len(recovery_array) >= minimum_count
                else None
            ),
            "1",
            status=("ok" if len(recovery_array) >= minimum_count else "not_computable"),
            reason=(
                None
                if len(recovery_array) >= minimum_count
                else "insufficient_phase_conditioned_rests_for_policy"
            ),
        )
        contributing_counts[f"{checkpoint_label}s"] = counts
        if len(values) < minimum_count:
            flags.append(f"insufficient_rests_reaching_{checkpoint_label}s")

    preceding_current = np.abs(_metric_values(rests, "preceding_load_median_current"))
    preceding_duration = _metric_values(rests, "preceding_load_duration")
    metrics["absolute_preceding_load_current_q50"] = _quantile_metric_with_unit(
        preceding_current,
        0.5,
        minimum_count=minimum_count,
        unit="A",
    )
    metrics["preceding_load_duration_q50"] = _quantile_metric_with_unit(
        preceding_duration,
        0.5,
        minimum_count=minimum_count,
        unit="s",
    )
    preceding_mode_counts = Counter(
        record["attributes"].get("preceding_mode") or "unavailable" for record in rests
    )
    member_summaries = []
    for record in rests:
        voltage_changes = {
            f"{seconds_label(checkpoint)}s": record["metrics"].get(
                f"voltage_change_at_{seconds_label(checkpoint)}s", metric(None, "V")
            )
            for checkpoint in config.relaxation_checkpoints_s
        }
        member_summaries.append(
            {
                "record_id": record["record_id"],
                "previous_phase": record["attributes"].get("previous_phase"),
                "preceding_mode": record["attributes"].get("preceding_mode"),
                "preceding_segment_id": record["attributes"].get("preceding_segment_id"),
                "preceding_load_median_current": record["metrics"].get(
                    "preceding_load_median_current", metric(None, "A")
                ),
                "preceding_load_duration": record["metrics"].get(
                    "preceding_load_duration", metric(None, "s")
                ),
                "rest_duration": record["metrics"].get("duration", metric(None, "s")),
                "temperature_baseline": record["metrics"].get(
                    "temperature_baseline", metric(None, "degC")
                ),
                "temperature_rise": record["metrics"].get("temperature_rise", metric(None, "degC")),
                "voltage_changes": voltage_changes,
            }
        )
    has_individual_response = bool(rests)

    return make_record(
        record_id=f"response.relaxation_signature:{cycle_id}",
        record_type="response.relaxation_signature",
        cell_id=cell_id,
        cycle_scope=cycle_id,
        source_intervals=_unique_intervals(source_intervals),
        attributes={
            "contributing_counts_by_previous_phase": contributing_counts,
            "member_record_ids": [record["record_id"] for record in rests],
            "member_summaries": member_summaries,
            "rejection_counts_by_reason": dict(sorted(rejection_counts.items())),
            "contributing_counts_by_preceding_mode": dict(sorted(preceding_mode_counts.items())),
            "recovery_sign_convention": (
                "-delta_v after charge; +delta_v after discharge; positive means relaxation toward the pre-load direction"
            ),
            "reference_frame": {
                "voltage_reference": "first_observed_rest_sample",
                "population": "rests_immediately_following_observed_charge_or_discharge",
                "preceding_phase_required": True,
                "equilibrium_assumed": False,
            },
            "confidence": {
                "level": (
                    "not_computable"
                    if not has_individual_response
                    else "medium"
                    if not any_phase_conditioned_sufficient
                    else "medium"
                    if flags
                    else "high"
                ),
                "basis": ["observed_rest", "fixed_time_checkpoints", *flags],
                "not_a_probability": True,
            },
        },
        metrics=metrics,
        provider="PyProBE+BFL",
        method_name="phase_conditioned_rest_voltage_recovery_v1",
        provider_version=f"PyProBE {pyprobe.__version__}; BFL 0.4.0",
        parameters={
            "analysis_policy_version": ANALYSIS_POLICY_VERSION,
            "checkpoints_s": list(config.relaxation_checkpoints_s),
            "minimum_rest_count": minimum_count,
        },
        references=[
            PYPROBE_REFERENCE,
            "https://doi.org/10.1038/s41467-022-29837-w",
        ],
        applicability_status=(
            "applicable"
            if any_phase_conditioned_sufficient
            else "partial"
            if has_individual_response
            else "not_computable"
        ),
        applicability_reasons=(
            []
            if any_phase_conditioned_sufficient
            else ["insufficient_phase_conditioned_rests_at_all_checkpoints"]
            if has_individual_response
            else ["no_rest_immediately_following_charge_or_discharge"]
        ),
        quality_status="warning" if flags else "ok",
        quality_flags=flags,
        interpretation_limits=[
            "Voltage recovery is a terminal response and is not an equilibrium-potential measurement.",
            "Recovery magnitude mixes ohmic, kinetic, diffusion, thermal, and state-window effects.",
            "Initial or otherwise unconditioned rests are excluded from response aggregates and remain available as individual rest evidence.",
        ],
    )


def analyze_current_steps(
    frame: pl.DataFrame,
    phases: list[dict[str, Any]],
    *,
    response_times_s: tuple[float, ...],
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
) -> list[dict[str, Any]]:
    """Extract protocol-neutral apparent delta-V/delta-I current-step responses."""

    selected = _select_cycle(frame, cycle_id)
    requested_times = tuple(sorted({2.0, *response_times_s}))
    required_time = max(requested_times)
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rest_min_s = config.policy("current_step_rest_min_s")
    pulse_max_s = config.policy("current_step_pulse_max_s")
    for index, pulse in enumerate(phases):
        if index == 0 or pulse["phase"] == "rest":
            continue
        rest = phases[index - 1]
        if (
            rest["phase"] == "rest"
            and rest["duration_s"] >= rest_min_s
            and required_time <= pulse["duration_s"] <= pulse_max_s
            and rest["duration_s"] >= pulse["duration_s"]
        ):
            candidates.append((rest, pulse))
    if not candidates:
        return [
            _unavailable_current_step_record(
                cell_id,
                cycle_id,
                "no_interval_passed_current_step_gate",
                record_id=f"response.current_step:{cycle_id}:unavailable",
            )
        ]
    return [
        _current_step_record(
            selected,
            rest,
            pulse,
            step_number=number,
            response_times_s=requested_times,
            config=config,
            cell_id=cell_id,
            cycle_id=cycle_id,
        )
        for number, (rest, pulse) in enumerate(candidates, start=1)
    ]


def analyze_current_step_summary(
    step_records: list[dict[str, Any]],
    *,
    response_times_s: tuple[float, ...],
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
) -> dict[str, Any]:
    """Summarize eligible current-step responses for compact consumers."""

    records = [
        record for record in step_records if record["record_type"] == "response.current_step"
    ]
    steps = [record for record in records if record["attributes"].get("candidate_detected", True)]
    computed = [record for record in steps if record["applicability"]["status"] == "applicable"]
    minimum_count = int(config.policy("current_step_summary_min_count"))
    metrics: dict[str, dict[str, Any]] = {
        "candidate_step_count": metric(len(steps), "1"),
        "computed_step_count": metric(len(computed), "1"),
        "rejected_step_count": metric(len(steps) - len(computed), "1"),
    }
    resistance_names = ["apparent_dc_resistance_first_valid"]
    resistance_names.extend(
        f"apparent_dc_resistance_{seconds_label(time_s)}s"
        for time_s in response_times_s
    )
    for name in resistance_names:
        values = _metric_values(computed, name, positive=True)
        metrics[f"{name}_valid_count"] = metric(len(values), "1")
        metrics[f"{name}_q50"] = _quantile_metric_with_unit(
            values, 0.5, minimum_count=minimum_count, unit="ohm"
        )
        metrics[f"{name}_q95"] = _quantile_metric_with_unit(
            values, 0.95, minimum_count=minimum_count, unit="ohm"
        )
        for direction in ("charge", "discharge"):
            directional = _metric_values(
                [
                    record
                    for record in computed
                    if record["attributes"].get("direction") == direction
                ],
                name,
                positive=True,
            )
            metrics[f"{name}_{direction}_q50"] = _quantile_metric_with_unit(
                directional, 0.5, minimum_count=minimum_count, unit="ohm"
            )
    metrics["pre_step_voltage_q05"] = _quantile_metric_with_unit(
        _metric_values(computed, "pre_step_voltage"),
        0.05,
        minimum_count=minimum_count,
        unit="V",
    )
    metrics["pre_step_voltage_q95"] = _quantile_metric_with_unit(
        _metric_values(computed, "pre_step_voltage"),
        0.95,
        minimum_count=minimum_count,
        unit="V",
    )
    metrics["absolute_delta_current_q50"] = _quantile_metric_with_unit(
        np.abs(_metric_values(computed, "delta_current")),
        0.5,
        minimum_count=minimum_count,
        unit="A",
    )
    metrics["pre_step_temperature_q05"] = _quantile_metric_with_unit(
        _metric_values(computed, "pre_step_temperature"),
        0.05,
        minimum_count=minimum_count,
        unit="degC",
    )
    metrics["pre_step_temperature_q95"] = _quantile_metric_with_unit(
        _metric_values(computed, "pre_step_temperature"),
        0.95,
        minimum_count=minimum_count,
        unit="degC",
    )
    direction_counts = {
        direction: sum(record["attributes"].get("direction") == direction for record in computed)
        for direction in ("charge", "discharge")
    }
    computed_flags = sorted({flag for record in computed for flag in record["quality"]["flags"]})
    rejection_counts = Counter(
        reason
        for record in steps
        if record["applicability"]["status"] != "applicable"
        for reason in record["applicability"]["reasons"]
    )
    enough = len(computed) >= minimum_count
    flags = list(computed_flags)
    if rejection_counts:
        flags.append("current_step_candidates_rejected")
    if not steps:
        flags.append("no_current_step_candidates")
    confidence_level = "not_computable"
    if enough:
        confidence_level = "medium" if rejection_counts or computed_flags else "high"
    return make_record(
        record_id=f"response.current_step_summary:{cycle_id}",
        record_type="response.current_step_summary",
        cell_id=cell_id,
        cycle_scope=cycle_id,
        source_intervals=_unique_intervals(
            [interval for record in computed for interval in record["source_intervals"]]
        ),
        attributes={
            "direction_counts": direction_counts,
            "member_record_ids": [record["record_id"] for record in computed],
            "rejection_counts_by_reason": dict(sorted(rejection_counts.items())),
            "reference_frame": {
                "current_sign": "charge-positive",
                "state_axis": "pre_step_voltage",
                "soc_reference": None,
            },
            "confidence": {
                "level": confidence_level,
                "basis": [
                    "fixed_response_times",
                    "rest_referenced_voltage",
                    "protocol_neutral_current_step_gate",
                    *flags,
                ],
                "not_a_probability": True,
            },
        },
        metrics=metrics,
        provider="BFL",
        method_name="current_step_distribution_v1",
        provider_version="0.4.0",
        parameters={
            "analysis_policy_version": ANALYSIS_POLICY_VERSION,
            "response_times_s": list(response_times_s),
            "summary_quantiles": [0.05, 0.5, 0.95],
            "minimum_step_count": minimum_count,
        },
        references=[
            "https://doi.org/10.1038/s41598-017-18424-5",
            "https://inldigitallibrary.inl.gov/content/uploads/50/2026/04/6308373.pdf",
        ],
        applicability_status="applicable" if enough else "not_computable",
        applicability_reasons=(
            [] if enough else ["insufficient_computable_current_steps_for_policy"]
        ),
        quality_status="warning" if flags else "ok",
        quality_flags=flags,
        interpretation_limits=[
            "The summary describes apparent terminal delta-V/delta-I response in the selected window.",
            "Comparisons require matched response time, current amplitude and direction, pre-step voltage, and temperature.",
            "No SOC, intrinsic resistance, health state, or named pulse protocol is inferred.",
        ],
    )


def _current_step_record(
    frame: pl.DataFrame,
    rest: dict[str, Any],
    pulse: dict[str, Any],
    *,
    step_number: int,
    response_times_s: tuple[float, ...],
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
) -> dict[str, Any]:
    baseline_window_s = config.policy("current_step_baseline_window_s")
    minimum_samples = int(config.policy("current_step_min_samples"))
    minimum_delta_current_a = config.policy("current_step_min_delta_current_a")
    noise_multiplier = config.policy("current_step_delta_current_noise_multiplier")
    baseline_rest_fraction_min = config.policy(
        "current_step_baseline_rest_fraction_min"
    )
    plateau_cv_max = config.policy("current_step_plateau_cv_max")
    plateau_match_tolerance = config.policy(
        "current_step_plateau_match_relative_tolerance"
    )
    high_confidence_cv_max = config.policy("current_step_high_confidence_cv_max")
    rest_frame = frame.slice(rest["start"], rest["end"] - rest["start"])
    pulse_frame = frame.slice(pulse["start"], pulse["end"] - pulse["start"])
    rest_time = rest_frame["test_time_s"].cast(pl.Float64).to_numpy()
    rest_current = rest_frame["current_a"].cast(pl.Float64).to_numpy()
    rest_voltage = rest_frame["voltage_v"].cast(pl.Float64).to_numpy()
    pulse_time = pulse_frame["test_time_s"].cast(pl.Float64).to_numpy()
    pulse_current = pulse_frame["current_a"].cast(pl.Float64).to_numpy()
    pulse_voltage = pulse_frame["voltage_v"].cast(pl.Float64).to_numpy()
    baseline_mask = (
        np.isfinite(rest_time)
        & np.isfinite(rest_current)
        & np.isfinite(rest_voltage)
        & (rest_time >= rest_time[-1] - baseline_window_s)
    )
    pulse_finite = np.isfinite(pulse_time) & np.isfinite(pulse_current) & np.isfinite(pulse_voltage)
    baseline_positions = np.flatnonzero(baseline_mask)
    pulse_positions = np.flatnonzero(pulse_finite)
    source_intervals = [
        _source_interval_for_positions(rest_frame, baseline_positions, role="pre_step_baseline"),
        _source_interval_for_positions(pulse_frame, pulse_positions, role="current_step_response"),
    ]
    reason = None
    if int(np.sum(baseline_mask)) < minimum_samples:
        reason = "insufficient_finite_baseline_samples_for_policy"
    elif int(np.sum(pulse_finite)) < minimum_samples:
        reason = "insufficient_finite_pulse_samples_for_policy"
    if reason is not None:
        return _unavailable_current_step_record(
            cell_id,
            cycle_id,
            reason,
            record_id=f"response.current_step:{cycle_id}:{step_number}",
            source_intervals=source_intervals,
        )

    baseline_current = float(np.median(rest_current[baseline_mask]))
    baseline_voltage = float(np.median(rest_voltage[baseline_mask]))
    baseline_noise = float(np.std(rest_current[baseline_mask]))
    baseline_rest_fraction = float(
        np.mean(np.abs(rest_current[baseline_mask]) <= config.rest_current_threshold_a)
    )
    temp_name = temperature_column(rest_frame)
    if temp_name is not None:
        baseline_temperature_values = (
            rest_frame[temp_name].cast(pl.Float64, strict=False).to_numpy()[baseline_mask]
        )
        baseline_temperature_values = baseline_temperature_values[
            np.isfinite(baseline_temperature_values)
        ]
    else:
        baseline_temperature_values = np.asarray([], dtype=float)
    plateau_current = float(np.median(pulse_current[pulse_finite]))
    delta_current = plateau_current - baseline_current
    plateau_cv = float(np.std(pulse_current[pulse_finite]) / max(abs(plateau_current), 1e-12))
    minimum_step = max(minimum_delta_current_a, noise_multiplier * baseline_noise)
    if baseline_rest_fraction < baseline_rest_fraction_min:
        reason = "pre_step_baseline_not_within_rest_current_threshold"
    elif abs(delta_current) <= minimum_step:
        reason = "delta_current_below_noise_gate"
    elif plateau_cv > plateau_cv_max:
        reason = "pulse_current_coefficient_of_variation_above_policy_limit"
    else:
        baseline_intervals = previous_intervals(
            rest_time[baseline_mask],
            rest_current[baseline_mask],
            sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
        )
        pulse_intervals = previous_intervals(
            pulse_time[pulse_finite],
            pulse_current[pulse_finite],
            sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
        )
        if (
            baseline_intervals.non_positive_interval_count
            or pulse_intervals.non_positive_interval_count
        ):
            reason = "non_positive_time_interval_in_current_step_window"
        elif (
            baseline_intervals.sampling_interval_outlier_count
            or pulse_intervals.sampling_interval_outlier_count
        ):
            reason = "sampling_interval_too_sparse_for_current_step"
        elif float(pulse_time[pulse_finite][-1] - pulse_time[pulse_finite][0]) < max(
            response_times_s
        ):
            reason = "finite_pulse_samples_do_not_reach_requested_time"
        else:
            reason = None
    if reason is not None:
        return _unavailable_current_step_record(
            cell_id,
            cycle_id,
            reason,
            record_id=f"response.current_step:{cycle_id}:{step_number}",
            source_intervals=source_intervals,
        )

    relative_time = pulse_time[pulse_finite] - pulse_time[pulse_finite][0]
    finite_current = pulse_current[pulse_finite]
    finite_voltage = pulse_voltage[pulse_finite]
    plateau_match = np.abs(finite_current - plateau_current) <= max(
        plateau_match_tolerance * abs(plateau_current), 1e-6
    )
    first_index = int(np.flatnonzero(plateau_match)[0])
    first_delta_voltage = float(finite_voltage[first_index] - baseline_voltage)
    first_delta_current = float(finite_current[first_index] - baseline_current)
    first_resistance = (
        first_delta_voltage / first_delta_current
        if abs(first_delta_current) > minimum_step
        else None
    )
    metrics = {
        "pre_step_voltage": metric(baseline_voltage, "V"),
        "pre_step_current": metric(baseline_current, "A"),
        "pre_step_voltage_std": metric(float(np.std(rest_voltage[baseline_mask])), "V"),
        "pre_step_current_std": metric(baseline_noise, "A"),
        "pre_step_rest_current_fraction": metric(baseline_rest_fraction, "1"),
        "pre_step_temperature": metric(
            json_number(np.median(baseline_temperature_values))
            if len(baseline_temperature_values)
            else None,
            "degC",
            reason=(
                None
                if len(baseline_temperature_values)
                else "no_finite_standardized_temperature_in_baseline"
            ),
        ),
        "pulse_plateau_current": metric(plateau_current, "A"),
        "pulse_current_coefficient_of_variation": metric(plateau_cv, "1"),
        "delta_current": metric(delta_current, "A"),
        "pulse_duration": metric(float(relative_time[-1]), "s"),
        "first_valid_latency": metric(float(relative_time[first_index]), "s"),
        "delta_voltage_first_valid": metric(first_delta_voltage, "V"),
        "delta_current_first_valid": metric(first_delta_current, "A"),
        "apparent_dc_resistance_first_valid": _resistance_metric(first_resistance),
    }
    resistance_at_time: dict[float, float | None] = {}
    for response_time in response_times_s:
        available = response_time <= relative_time[-1]
        if available:
            insertion = int(np.searchsorted(relative_time, response_time, side="left"))
            if insertion < len(relative_time) and np.isclose(
                relative_time[insertion], response_time
            ):
                bracket_width = 0.0
                nearest_offset = abs(float(relative_time[insertion] - response_time))
            else:
                left = max(insertion - 1, 0)
                right = min(insertion, len(relative_time) - 1)
                bracket_width = float(relative_time[right] - relative_time[left])
                nearest_offset = float(
                    min(
                        abs(relative_time[left] - response_time),
                        abs(relative_time[right] - response_time),
                    )
                )
            voltage_at_time = float(np.interp(response_time, relative_time, finite_voltage))
            current_at_time = float(np.interp(response_time, relative_time, finite_current))
            delta_voltage = voltage_at_time - baseline_voltage
            delta_i = current_at_time - baseline_current
            resistance = delta_voltage / delta_i if abs(delta_i) > minimum_step else None
        else:
            delta_voltage = None
            resistance = None
            bracket_width = None
            nearest_offset = None
        resistance_at_time[response_time] = resistance
        suffix = f"{seconds_label(response_time)}s"
        metrics[f"delta_voltage_{suffix}"] = metric(
            delta_voltage,
            "V",
            status="ok" if delta_voltage is not None else "not_computable",
            reason=None if delta_voltage is not None else "pulse_does_not_reach_requested_time",
        )
        metrics[f"delta_current_{suffix}"] = metric(
            delta_i if available else None,
            "A",
            status="ok" if available else "not_computable",
            reason=None if available else "pulse_does_not_reach_requested_time",
        )
        metrics[f"apparent_dc_resistance_{suffix}"] = _resistance_metric(
            resistance,
            reason=None if available else "pulse_does_not_reach_requested_time",
        )
        metrics[f"checkpoint_bracket_width_{suffix}"] = metric(
            bracket_width,
            "s",
            status="ok" if bracket_width is not None else "not_computable",
            reason=(None if bracket_width is not None else "pulse_does_not_reach_requested_time"),
        )
        metrics[f"checkpoint_nearest_sample_offset_{suffix}"] = metric(
            nearest_offset,
            "s",
            status="ok" if nearest_offset is not None else "not_computable",
            reason=(None if nearest_offset is not None else "pulse_does_not_reach_requested_time"),
        )
    longest_time = max(response_times_s)
    longest_resistance = resistance_at_time[longest_time]
    metrics[f"polarization_growth_to_{seconds_label(longest_time)}s"] = metric(
        (
            longest_resistance - first_resistance
            if longest_resistance is not None and first_resistance is not None
            else None
        ),
        "ohm",
        status=(
            "ok"
            if longest_resistance is not None and first_resistance is not None
            else "not_computable"
        ),
        reason=(
            None
            if longest_resistance is not None and first_resistance is not None
            else "first and requested-time resistance are not both available"
        ),
    )
    confidence = "high" if plateau_cv <= high_confidence_cv_max else "medium"
    direction = "charge" if delta_current > 0 else "discharge"
    return make_record(
        record_id=f"response.current_step:{cycle_id}:{step_number}",
        record_type="response.current_step",
        cell_id=cell_id,
        cycle_scope=cycle_id,
        source_intervals=source_intervals,
        attributes={
            "step_number": step_number,
            "candidate_detected": True,
            "direction": direction,
            "preceding_phase": rest["phase"],
            "following_phase": pulse.get("next_phase"),
            "temperature_column": temp_name,
            "reference_frame": {
                "voltage_baseline": "median_of_configured_final_observed_rest_window",
                "current_sign": "charge-positive",
                "state_axis": "pre_step_voltage",
                "soc_reference": None,
            },
            "confidence": {
                "level": confidence,
                "basis": [
                    "finite_configured_rest_baseline_window",
                    "delta_current_above_observed_baseline_noise",
                    "pulse_plateau_cv_within_policy_limit",
                    "requested_times_bracketed_by_finite_pulse_samples",
                ],
                "not_a_probability": True,
            },
        },
        metrics=metrics,
        provider="BFL",
        method_name="current_step_delta_v_over_delta_i_v1",
        provider_version="0.4.0",
        parameters={
            "analysis_policy_version": ANALYSIS_POLICY_VERSION,
            "baseline_window_s": baseline_window_s,
            "minimum_baseline_samples": minimum_samples,
            "minimum_pulse_samples": minimum_samples,
            "minimum_delta_current_a": minimum_delta_current_a,
            "minimum_delta_current_noise_multiplier": noise_multiplier,
            "minimum_baseline_rest_current_fraction": baseline_rest_fraction_min,
            "pulse_current_cv_maximum": plateau_cv_max,
            "pulse_plateau_match_relative_tolerance": plateau_match_tolerance,
            "high_confidence_pulse_current_cv_maximum": high_confidence_cv_max,
            "response_times_s": list(response_times_s),
            "checkpoint_method": "linear_interpolation_within_observed_pulse",
        },
        references=[
            "https://doi.org/10.1038/s41598-017-18424-5",
            "https://inldigitallibrary.inl.gov/content/uploads/50/2026/04/6308373.pdf",
        ],
        quality_status="warning" if confidence == "medium" else "ok",
        quality_flags=["pulse_current_variation"] if confidence == "medium" else [],
        interpretation_limits=[
            "This is an apparent terminal delta-V/delta-I response, not intrinsic resistance.",
            "SOC and a named HPPC or GITT protocol are not inferred.",
            "Resistance depends on response time, acquisition rate, current amplitude, pre-step voltage, and temperature.",
        ],
    )


def _select_cycle(frame: pl.DataFrame, cycle_id: int | None) -> pl.DataFrame:
    if cycle_id is None or "cycle_index" not in frame.columns:
        return frame.sort("test_time_s")
    return frame.filter(pl.col("cycle_index") == cycle_id).sort("test_time_s")


def _sum(values: np.ndarray) -> float:
    return float(np.sum(values)) if len(values) else 0.0


def _quantile_metric(values: np.ndarray, quantile: float, *, minimum_count: int) -> dict[str, Any]:
    enough = len(values) >= minimum_count
    return metric(
        json_number(np.quantile(values, quantile)) if enough else None,
        "V",
        status="ok" if enough else "not_computable",
        reason=None if enough else f"fewer_than_{minimum_count}_eligible_rests",
    )


def _quantile_metric_with_unit(
    values: np.ndarray,
    quantile: float,
    *,
    minimum_count: int,
    unit: str,
) -> dict[str, Any]:
    enough = len(values) >= minimum_count
    return metric(
        json_number(np.quantile(values, quantile)) if enough else None,
        unit,
        status="ok" if enough else "not_computable",
        reason=None if enough else f"fewer_than_{minimum_count}_eligible_values",
    )


def _metric_values(
    records: list[dict[str, Any]], name: str, *, positive: bool = False
) -> np.ndarray:
    values = np.asarray(
        [
            record["metrics"][name]["value"]
            for record in records
            if name in record["metrics"] and record["metrics"][name]["value"] is not None
        ],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    return values[values > 0] if positive else values


def _resistance_metric(value: float | None, *, reason: str | None = None) -> dict[str, Any]:
    valid = value is not None and np.isfinite(value) and value > 0
    return metric(
        value if valid else None,
        "ohm",
        status="ok" if valid else "not_computable",
        reason=reason or (None if valid else "non_positive_or_unavailable_apparent_resistance"),
    )


def _unique_intervals(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for interval in intervals:
        key = (
            interval.get("start_row"),
            interval.get("end_row"),
            interval.get("start_time_s"),
            interval.get("end_time_s"),
        )
        if key not in seen:
            seen.add(key)
            output.append(interval)
    return output


def _source_interval_for_positions(
    frame: pl.DataFrame,
    positions: np.ndarray,
    *,
    role: str,
) -> dict[str, Any]:
    """Map exact calculation samples back to source rows, records, and time."""

    used_candidate_bounds = not len(positions)
    if used_candidate_bounds:
        positions = np.asarray([0, frame.height - 1], dtype=int)
    start = int(positions[0])
    end = int(positions[-1])
    rows = frame["_source_row"].to_numpy()
    records = (
        frame.select(
            pl.coalesce(
                pl.col("record_index").cast(pl.Int64, strict=False),
                pl.col("_source_row").cast(pl.Int64),
            ).alias("_source_record")
        )["_source_record"].to_numpy()
        if "record_index" in frame.columns
        else rows
    )
    times = frame["test_time_s"].cast(pl.Float64).to_numpy()
    output = {
        "role": role,
        "start_row": int(rows[start]),
        "end_row": int(rows[end]),
        "start_record": int(records[start]),
        "end_record": int(records[end]),
        "start_time_s": float(times[start]),
        "end_time_s": float(times[end]),
    }
    if used_candidate_bounds:
        output["selection"] = "candidate_bounds_no_finite_calculation_samples"
    return output


def _unavailable_directional_record(
    cell_id: str, cycle_id: int | None, reason: str
) -> dict[str, Any]:
    return make_record(
        record_id=f"response.directional_energy_summary:{cycle_id}",
        record_type="response.directional_energy_summary",
        cell_id=cell_id,
        cycle_scope=cycle_id,
        source_intervals=[],
        provider="BFL",
        method_name="directional_previous_zoh_energy_voltage_v1",
        provider_version="0.4.0",
        applicability_status="not_computable",
        applicability_reasons=[reason],
        quality_status="warning",
        quality_flags=["input_not_eligible"],
        interpretation_limits=[
            "No directional response was inferred without eligible measurements."
        ],
    )


def _unavailable_current_step_record(
    cell_id: str,
    cycle_id: int | None,
    reason: str,
    *,
    record_id: str,
    source_intervals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return make_record(
        record_id=record_id,
        record_type="response.current_step",
        cell_id=cell_id,
        cycle_scope=cycle_id,
        source_intervals=source_intervals or [],
        attributes={
            "candidate_detected": bool(source_intervals),
            "reference_frame": {
                "current_sign": "charge-positive",
                "state_axis": "pre_step_voltage",
                "soc_reference": None,
            },
            "confidence": {
                "level": "not_computable",
                "basis": [reason],
                "not_a_probability": True,
            },
        },
        provider="BFL",
        method_name="current_step_delta_v_over_delta_i_v1",
        provider_version="0.4.0",
        references=["https://doi.org/10.1038/s41598-017-18424-5"],
        applicability_status="not_computable",
        applicability_reasons=[reason],
        quality_status="warning",
        quality_flags=["input_not_eligible"],
        interpretation_limits=[
            "No apparent resistance was emitted when the declared current-step gates failed."
        ],
    )
