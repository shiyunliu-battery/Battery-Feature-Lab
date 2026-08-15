"""Capacity-aligned charge/discharge voltage response profiles."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from battery_feature_lab.analysis.adapters import temperature_column
from battery_feature_lab.analysis.numerics import json_number
from battery_feature_lab.analysis.schema import (
    ANALYSIS_POLICY_VERSION,
    AnalysisConfig,
    make_record,
    metric,
)


def _merge_direction_runs(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for phase in phases:
        if phase.get("phase") not in {"charge", "discharge"}:
            continue
        if runs and runs[-1]["phase"] == phase["phase"] and runs[-1]["end"] == phase["start"]:
            runs[-1]["end"] = phase["end"]
            runs[-1]["segments"].append(phase)
            runs[-1]["source_intervals"].append(phase["source_interval"])
        else:
            runs.append(
                {
                    "phase": phase["phase"],
                    "start": phase["start"],
                    "end": phase["end"],
                    "segments": [phase],
                    "source_intervals": [phase["source_interval"]],
                }
            )
    for run in runs:
        first_index = int(run["segments"][0]["segment_index"])
        last_index = int(run["segments"][-1]["segment_index"])
        previous = phases[first_index - 1] if first_index else None
        following = phases[last_index + 1] if last_index + 1 < len(phases) else None
        run["preceding_rest"] = previous if previous and previous["phase"] == "rest" else None
        run["following_rest"] = following if following and following["phase"] == "rest" else None
        run["preceding_rest_s"] = previous["duration_s"] if run["preceding_rest"] else 0.0
        run["following_rest_s"] = following["duration_s"] if run["following_rest"] else 0.0
    return runs


def _threshold_text(value: float) -> str:
    """Format a policy threshold while preserving legacy default reason strings."""

    return f"{value:g}"


def _run_profile(
    frame: pl.DataFrame,
    run: dict[str, Any],
    *,
    config: AnalysisConfig,
) -> dict[str, Any]:
    boundary_count = run["end"] - run["start"] + (run["end"] < frame.height)
    sub = frame.slice(run["start"], boundary_count)
    time = sub["test_time_s"].cast(pl.Float64).to_numpy()
    current = sub["current_a"].cast(pl.Float64).to_numpy().copy()
    voltage = sub["voltage_v"].cast(pl.Float64).to_numpy().copy()
    if run["end"] < frame.height and len(voltage) >= 2:
        current[-1] = current[-2]
        voltage[-1] = voltage[-2]
    dt = np.diff(time)
    valid = np.isfinite(dt) & (dt > 0) & np.isfinite(current[:-1]) & np.isfinite(voltage[:-1])
    dq = np.where(valid, np.abs(current[:-1]) * dt / 3600.0, 0.0)
    cumulative = np.r_[0.0, np.cumsum(dq)]
    capacity = float(cumulative[-1])
    finite_voltage = voltage[np.isfinite(voltage)]
    expected_increasing = run["phase"] == "charge"
    if len(finite_voltage) >= 3:
        dv = np.diff(finite_voltage)
        tolerance = max(1e-5, 0.001 * float(np.ptp(finite_voltage)))
        monotone_fraction = float(
            np.mean(dv >= -tolerance) if expected_increasing else np.mean(dv <= tolerance)
        )
    else:
        monotone_fraction = 0.0
    total_duration = float(np.sum(dt[np.isfinite(dt) & (dt > 0)]))
    covered_duration = float(np.sum(dt[valid]))
    coverage = covered_duration / total_duration if total_duration else 0.0
    finite_current = current[np.isfinite(current)]
    temp_name = temperature_column(sub)
    finite_temperature = (
        sub[temp_name].cast(pl.Float64, strict=False).to_numpy()
        if temp_name is not None
        else np.asarray([], dtype=float)
    )
    finite_temperature = finite_temperature[np.isfinite(finite_temperature)]
    minimum_samples = int(config.policy("profile_min_samples"))
    minimum_duration_s = config.policy("profile_min_duration_s")
    minimum_coverage = config.policy("profile_finite_coverage_min")
    minimum_direction_consistency = config.policy("profile_direction_consistency_min")
    minimum_bracketing_rest_s = config.policy("profile_bracketing_rest_min_s")
    reasons: list[str] = []
    if len(sub) < minimum_samples:
        reasons.append(f"fewer_than_{minimum_samples}_samples")
    if total_duration < minimum_duration_s:
        reasons.append(f"duration_below_{_threshold_text(minimum_duration_s)}s")
    if coverage < minimum_coverage:
        reasons.append(
            f"finite_interval_coverage_below_{_threshold_text(minimum_coverage)}"
        )
    if capacity <= 0:
        reasons.append("non_positive_phase_capacity")
    expected_sign = 1 if expected_increasing else -1
    if (
        not len(finite_current)
        or float(np.mean(np.sign(finite_current) == expected_sign))
        < minimum_direction_consistency
    ):
        percentage = _threshold_text(100.0 * minimum_direction_consistency)
        reasons.append(f"current_direction_not_{percentage}_percent_consistent")
    if (
        run["preceding_rest_s"] < minimum_bracketing_rest_s
        or run["following_rest_s"] < minimum_bracketing_rest_s
    ):
        rest_seconds = _threshold_text(minimum_bracketing_rest_s)
        reasons.append(f"directional_block_not_bracketed_by_{rest_seconds}s_rests")
    return {
        **run,
        "time": time,
        "current": current,
        "voltage": voltage,
        "capacity_axis": cumulative,
        "capacity_ah": capacity,
        "duration_s": total_duration,
        "coverage": coverage,
        "voltage_monotone_fraction": monotone_fraction,
        "start_voltage_v": float(finite_voltage[0]) if len(finite_voltage) else None,
        "end_voltage_v": float(finite_voltage[-1]) if len(finite_voltage) else None,
        "mean_abs_current_a": (
            float(np.sum(np.abs(current[:-1][valid]) * dt[valid]) / covered_duration)
            if covered_duration
            else None
        ),
        "median_temperature_deg_c": (
            float(np.median(finite_temperature)) if len(finite_temperature) else None
        ),
        "reasons": reasons,
    }


def _interp_unique(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]
    order = np.argsort(x_valid, kind="mergesort")
    x_valid = x_valid[order]
    y_valid = y_valid[order]
    unique, indexes = np.unique(x_valid, return_index=True)
    return np.interp(grid, unique, y_valid[indexes])


def _candidate_intervals(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for profile in profiles:
        preceding = profile.get("preceding_rest")
        following = profile.get("following_rest")
        members = [
            (preceding, "leading_rest"),
            *[(segment, f"{profile['phase']}_member") for segment in profile["segments"]],
            (following, "trailing_rest"),
        ]
        for segment, role in members:
            if segment is None:
                continue
            source = dict(segment["source_interval"])
            key = (int(source["start_row"]), int(source["end_row"]), role)
            if key in seen:
                continue
            seen.add(key)
            source["role"] = role
            intervals.append(source)
    return intervals


def _share_intermediate_rest(charge: dict[str, Any], discharge: dict[str, Any]) -> bool:
    """Return whether two opposite-direction blocks are consecutive around one rest."""

    if charge["end"] < discharge["start"]:
        return charge.get("following_rest") is not None and charge.get(
            "following_rest"
        ) is discharge.get("preceding_rest")
    return discharge.get("following_rest") is not None and discharge.get(
        "following_rest"
    ) is charge.get("preceding_rest")


def analyze_capacity_aligned_profile(
    frame: pl.DataFrame,
    phases: list[dict[str, Any]],
    *,
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
    cycle_id_source: str,
) -> dict[str, Any]:
    """Compare complete, state-window-matched charge and discharge trajectories."""

    grid_points = int(config.policy("profile_grid_points"))
    endpoint_tolerance_v = config.policy("profile_endpoint_tolerance_v")
    minimum_capacity_balance = config.policy("profile_capacity_balance_min")
    minimum_voltage_span_overlap = config.policy("profile_voltage_span_overlap_min")
    minimum_current_ratio = config.policy("profile_current_ratio_min")
    maximum_current_ratio = config.policy("profile_current_ratio_max")
    minimum_current_match = config.policy("profile_current_match_fraction_min")
    maximum_voltage_wrong_way = config.policy("profile_voltage_wrong_way_fraction_max")
    maximum_temperature_difference = config.policy(
        "profile_temperature_difference_max_deg_c"
    )
    method_parameters: dict[str, Any] = {
        "analysis_policy_version": ANALYSIS_POLICY_VERSION,
        "integration": "previous_zoh",
        "grid_points": grid_points,
        "minimum_samples": int(config.policy("profile_min_samples")),
        "minimum_duration_s": config.policy("profile_min_duration_s"),
        "minimum_finite_interval_coverage": config.policy(
            "profile_finite_coverage_min"
        ),
        "minimum_current_direction_consistency_fraction": config.policy(
            "profile_direction_consistency_min"
        ),
        "minimum_bracketing_rest_duration_s": config.policy(
            "profile_bracketing_rest_min_s"
        ),
        "endpoint_tolerance_v": endpoint_tolerance_v,
        "minimum_capacity_balance": minimum_capacity_balance,
        "minimum_voltage_span_overlap_fraction": minimum_voltage_span_overlap,
        "current_ratio_range": [minimum_current_ratio, maximum_current_ratio],
        "minimum_current_match_fraction": minimum_current_match,
        "maximum_voltage_wrong_way_fraction": maximum_voltage_wrong_way,
        "maximum_median_temperature_difference_deg_c": maximum_temperature_difference,
    }
    profiles = [
        _run_profile(frame, run, config=config) for run in _merge_direction_runs(phases)
    ]
    charges = [item for item in profiles if item["phase"] == "charge" and not item["reasons"]]
    discharges = [item for item in profiles if item["phase"] == "discharge" and not item["reasons"]]
    pair_candidates: list[dict[str, Any]] = []
    for charge in charges:
        for discharge in discharges:
            if not _share_intermediate_rest(charge, discharge):
                continue
            low_mismatch = abs(charge["start_voltage_v"] - discharge["end_voltage_v"])
            high_mismatch = abs(charge["end_voltage_v"] - discharge["start_voltage_v"])
            balance = min(charge["capacity_ah"], discharge["capacity_ah"]) / max(
                charge["capacity_ah"], discharge["capacity_ah"]
            )
            charge_span = abs(charge["end_voltage_v"] - charge["start_voltage_v"])
            discharge_span = abs(discharge["start_voltage_v"] - discharge["end_voltage_v"])
            overlap = max(
                min(charge["end_voltage_v"], discharge["start_voltage_v"])
                - max(charge["start_voltage_v"], discharge["end_voltage_v"]),
                0.0,
            )
            voltage_span_overlap = overlap / max(charge_span, discharge_span, 1e-12)
            common_capacity = min(charge["capacity_ah"], discharge["capacity_ah"])
            probe_grid = np.linspace(0.0, common_capacity, grid_points)
            charge_upper_axis = charge["capacity_ah"] - charge["capacity_axis"]
            discharge_upper_axis = discharge["capacity_axis"]
            charge_current = np.abs(
                _interp_unique(charge_upper_axis, charge["current"], probe_grid)
            )
            discharge_current = np.abs(
                _interp_unique(discharge_upper_axis, discharge["current"], probe_grid)
            )
            ratio = np.divide(
                charge_current,
                discharge_current,
                out=np.full_like(charge_current, np.inf),
                where=discharge_current > 1e-12,
            )
            current_match_fraction = float(
                np.mean(
                    (ratio >= minimum_current_ratio) & (ratio <= maximum_current_ratio)
                )
            )
            charge_probe_voltage = _interp_unique(charge_upper_axis, charge["voltage"], probe_grid)
            discharge_probe_voltage = _interp_unique(
                discharge_upper_axis, discharge["voltage"], probe_grid
            )

            def wrong_way_fraction(values: np.ndarray) -> float:
                differences = np.diff(values)
                total_variation = float(np.sum(np.abs(differences)))
                return (
                    float(np.sum(np.clip(differences, 0.0, None)) / total_variation)
                    if total_variation > 0
                    else 0.0
                )

            charge_wrong_way = wrong_way_fraction(charge_probe_voltage)
            discharge_wrong_way = wrong_way_fraction(discharge_probe_voltage)
            charge_temperature = charge["median_temperature_deg_c"]
            discharge_temperature = discharge["median_temperature_deg_c"]
            temperature_difference = (
                abs(charge_temperature - discharge_temperature)
                if charge_temperature is not None and discharge_temperature is not None
                else None
            )
            reasons: list[str] = []
            if low_mismatch > endpoint_tolerance_v:
                reasons.append(
                    "low_voltage_endpoint_mismatch_above_"
                    f"{_threshold_text(endpoint_tolerance_v)}V"
                )
            if high_mismatch > endpoint_tolerance_v:
                reasons.append(
                    "high_voltage_endpoint_mismatch_above_"
                    f"{_threshold_text(endpoint_tolerance_v)}V"
                )
            if balance < minimum_capacity_balance:
                reasons.append(
                    "paired_capacity_balance_below_"
                    f"{_threshold_text(minimum_capacity_balance)}"
                )
            if voltage_span_overlap < minimum_voltage_span_overlap:
                reasons.append("charge_voltage_state_span_does_not_cover_discharge_span")
            if current_match_fraction < minimum_current_match:
                reasons.append(
                    "capacity_weighted_current_match_fraction_below_"
                    f"{minimum_current_match:.2f}"
                )
            if (
                charge_wrong_way > maximum_voltage_wrong_way
                or discharge_wrong_way > maximum_voltage_wrong_way
            ):
                reasons.append(
                    "capacity_aligned_voltage_wrong_way_fraction_above_"
                    f"{_threshold_text(maximum_voltage_wrong_way)}"
                )
            if (
                temperature_difference is not None
                and temperature_difference > maximum_temperature_difference
            ):
                reasons.append(
                    "paired_median_temperature_difference_above_"
                    f"{_threshold_text(maximum_temperature_difference)}degC"
                )
            pair_candidates.append(
                {
                    "charge": charge,
                    "discharge": discharge,
                    "low_mismatch_v": low_mismatch,
                    "high_mismatch_v": high_mismatch,
                    "capacity_balance": balance,
                    "voltage_span_overlap_fraction": voltage_span_overlap,
                    "current_match_fraction": current_match_fraction,
                    "charge_voltage_wrong_way_fraction": charge_wrong_way,
                    "discharge_voltage_wrong_way_fraction": discharge_wrong_way,
                    "median_temperature_difference_deg_c": temperature_difference,
                    "reasons": reasons,
                    "score": low_mismatch + high_mismatch + (1.0 - balance),
                }
            )
    eligible = [item for item in pair_candidates if not item["reasons"]]
    selected = min(eligible, key=lambda item: item["score"]) if eligible else None
    best_candidate = (
        min(pair_candidates, key=lambda item: item["score"]) if pair_candidates else None
    )
    source_intervals = _candidate_intervals(profiles)
    candidate_attributes = [
        {
            "phase": item["phase"],
            "capacity_ah": item["capacity_ah"],
            "duration_s": item["duration_s"],
            "start_voltage_v": item["start_voltage_v"],
            "end_voltage_v": item["end_voltage_v"],
            "finite_interval_coverage": item["coverage"],
            "voltage_monotone_fraction": item["voltage_monotone_fraction"],
            "median_temperature_deg_c": item["median_temperature_deg_c"],
            "eligibility_reasons": item["reasons"],
            "source_step_indices": [
                segment.get("source_step_index") for segment in item["segments"]
            ],
        }
        for item in profiles
    ]
    if selected is None:
        reasons = sorted({reason for item in pair_candidates for reason in item["reasons"]})
        if not charges:
            reasons.append("no_eligible_charge_trajectory")
        if not discharges:
            reasons.append("no_eligible_discharge_trajectory")
        if charges and discharges and not pair_candidates:
            reasons.append("no_temporally_adjacent_charge_discharge_pair")
        reasons = sorted(set(reasons)) or ["no_capacity_aligned_pair"]
        return make_record(
            record_id=f"response.capacity_aligned_profile:{cycle_id}",
            record_type="response.capacity_aligned_profile",
            cell_id=cell_id,
            cycle_scope=cycle_id,
            source_intervals=source_intervals,
            attributes={
                "candidate_trajectories": candidate_attributes,
                "pairing": {
                    "status": "unmatched",
                    "endpoint_tolerance_v": endpoint_tolerance_v,
                    "minimum_capacity_balance": minimum_capacity_balance,
                    "minimum_voltage_span_overlap_fraction": minimum_voltage_span_overlap,
                    "minimum_current_match_fraction": minimum_current_match,
                },
                "cycle_id_source": cycle_id_source,
                "reference_frame": {
                    "capacity_axis": "capacity_from_shared_upper_endpoint_ah",
                    "state_window_matched": False,
                    "is_soc": False,
                },
            },
            metrics={
                "eligible_charge_trajectory_count": metric(len(charges), "1"),
                "eligible_discharge_trajectory_count": metric(len(discharges), "1"),
                "eligible_pair_count": metric(0, "1"),
                "candidate_charge_capacity": metric(
                    best_candidate["charge"]["capacity_ah"] if best_candidate else None,
                    "Ah",
                ),
                "candidate_discharge_capacity": metric(
                    best_candidate["discharge"]["capacity_ah"] if best_candidate else None,
                    "Ah",
                ),
                "candidate_capacity_balance": metric(
                    best_candidate["capacity_balance"] if best_candidate else None,
                    "1",
                ),
                "candidate_unpaired_capacity": metric(
                    (
                        abs(
                            best_candidate["charge"]["capacity_ah"]
                            - best_candidate["discharge"]["capacity_ah"]
                        )
                        if best_candidate
                        else None
                    ),
                    "Ah",
                ),
                "candidate_voltage_span_overlap_fraction": metric(
                    best_candidate["voltage_span_overlap_fraction"] if best_candidate else None,
                    "1",
                ),
                "candidate_current_match_fraction": metric(
                    best_candidate["current_match_fraction"] if best_candidate else None,
                    "1",
                ),
            },
            provider="BFL",
            method_name="capacity_aligned_bidirectional_profile_v1",
            provider_version="0.4.0",
            parameters=method_parameters,
            references=["https://doi.org/10.1038/s42256-024-00972-x"],
            applicability_status="not_computable",
            applicability_reasons=reasons,
            quality_status=("ok" if temperature_column(frame) else "warning"),
            quality_flags=(
                [] if temperature_column(frame) else ["standardized_temperature_unavailable"]
            ),
            interpretation_limits=[
                "No voltage-gap or polarization curve is emitted when charge and discharge state windows are not demonstrably comparable.",
                "Endpoint matching is a structural completeness gate, not proof of equilibrium or identical temperature history.",
            ],
        )

    charge = selected["charge"]
    discharge = selected["discharge"]
    common_capacity = min(charge["capacity_ah"], discharge["capacity_ah"])
    grid = np.linspace(0.0, common_capacity, grid_points)
    charge_axis = charge["capacity_ah"] - charge["capacity_axis"]
    discharge_axis = discharge["capacity_axis"]
    charge_voltage = _interp_unique(charge_axis, charge["voltage"], grid)
    discharge_voltage = _interp_unique(discharge_axis, discharge["voltage"], grid)
    charge_current = np.abs(_interp_unique(charge_axis, charge["current"], grid))
    discharge_current = np.abs(_interp_unique(discharge_axis, discharge["current"], grid))
    delta_voltage = charge_voltage - discharge_voltage
    current_sum = charge_current + discharge_current
    apparent_polarization = np.divide(
        delta_voltage,
        current_sum,
        out=np.full(grid_points, np.nan),
        where=current_sum > 1e-12,
    )
    voltage_gap_integral = float(
        np.sum(0.5 * (delta_voltage[1:] + delta_voltage[:-1]) * np.diff(grid))
    )
    return make_record(
        record_id=f"response.capacity_aligned_profile:{cycle_id}",
        record_type="response.capacity_aligned_profile",
        cell_id=cell_id,
        cycle_scope=cycle_id,
        source_intervals=_candidate_intervals([charge, discharge]),
        attributes={
            "candidate_trajectories": candidate_attributes,
            "pairing": {
                "status": "matched",
                "endpoint_tolerance_v": endpoint_tolerance_v,
                "minimum_capacity_balance": minimum_capacity_balance,
                "minimum_voltage_span_overlap_fraction": minimum_voltage_span_overlap,
                "minimum_current_match_fraction": minimum_current_match,
                "low_voltage_endpoint_mismatch_v": selected["low_mismatch_v"],
                "high_voltage_endpoint_mismatch_v": selected["high_mismatch_v"],
                "voltage_span_overlap_fraction": selected["voltage_span_overlap_fraction"],
                "current_match_fraction": selected["current_match_fraction"],
            },
            "cycle_id_source": cycle_id_source,
            "selected_charge_step_indices": [
                segment.get("source_step_index") for segment in charge["segments"]
            ],
            "selected_discharge_step_indices": [
                segment.get("source_step_index") for segment in discharge["segments"]
            ],
            "reference_frame": {
                "capacity_axis": "capacity_from_shared_upper_endpoint_ah",
                "state_window_matched": True,
                "is_soc": False,
                "current_sign": "charge-positive",
            },
        },
        metrics={
            "charge_capacity": metric(charge["capacity_ah"], "Ah"),
            "discharge_capacity": metric(discharge["capacity_ah"], "Ah"),
            "aligned_capacity": metric(common_capacity, "Ah"),
            "directional_capacity_balance": metric(selected["capacity_balance"], "1"),
            "charge_mean_absolute_current": metric(charge["mean_abs_current_a"], "A"),
            "discharge_mean_absolute_current": metric(discharge["mean_abs_current_a"], "A"),
            "voltage_gap_q50": metric(json_number(np.quantile(delta_voltage, 0.5)), "V"),
            "voltage_gap_q95": metric(json_number(np.quantile(delta_voltage, 0.95)), "V"),
            "capacity_weighted_mean_voltage_gap": metric(
                json_number(voltage_gap_integral / common_capacity),
                "V",
            ),
            "voltage_gap_integral": metric(voltage_gap_integral, "Wh"),
            "apparent_paired_polarization_q50": metric(
                json_number(np.quantile(apparent_polarization, 0.5)), "ohm"
            ),
            "profile_point_count": metric(grid_points, "1"),
        },
        series={
            "capacity_from_shared_upper_endpoint_ah": grid.tolist(),
            "capacity_fraction": (grid / common_capacity).tolist(),
            "charge_voltage_v": charge_voltage.tolist(),
            "discharge_voltage_v": discharge_voltage.tolist(),
            "charge_current_a": charge_current.tolist(),
            "discharge_current_a": discharge_current.tolist(),
            "voltage_gap_v": delta_voltage.tolist(),
            "apparent_paired_polarization_ohm": apparent_polarization.tolist(),
        },
        provider="BFL",
        method_name="capacity_aligned_bidirectional_profile_v1",
        provider_version="0.4.0",
        parameters={
            **method_parameters,
            "interpolation": "linear_on_monotone_capacity",
        },
        references=["https://doi.org/10.1038/s42256-024-00972-x"],
        quality_status=("ok" if temperature_column(frame) else "warning"),
        quality_flags=(
            [] if temperature_column(frame) else ["standardized_temperature_unavailable"]
        ),
        interpretation_limits=[
            "The paired voltage gap combines hysteresis, ohmic, kinetic, diffusion, thermal, and timing effects.",
            "Apparent paired polarization divides the voltage gap by pointwise capacity-aligned current magnitudes; it is not intrinsic resistance.",
            "Comparable endpoints and capacity do not establish equilibrium or identical thermal history.",
        ],
    )
