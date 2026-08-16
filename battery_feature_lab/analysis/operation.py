"""Operation phase, mode, and exposure analysis."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl
import pyprobe

from battery_feature_lab.analysis.adapters import pyprobe_experiment, temperature_column
from battery_feature_lab.analysis.numerics import (
    integrate_previous,
    json_number,
    previous_intervals,
    weighted_quantile,
    weighted_summary,
)
from battery_feature_lab.analysis.schema import (
    ANALYSIS_POLICY_VERSION,
    DEFAULT_ANALYSIS_POLICY,
    AnalysisConfig,
    make_record,
    metric,
)

PYPROBE_REFERENCE = "https://doi.org/10.21105/joss.07474"


def analyze_operation(
    frame: pl.DataFrame,
    pyprobe_frame: pl.DataFrame | None,
    *,
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
    provider_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build representative-cycle operation records and return segment metadata."""

    selected = _select_cycle(frame, cycle_id)
    selected_pyprobe = _select_pyprobe_cycle(pyprobe_frame, cycle_id)
    mode_policy = _mode_policy_values(config)
    phases = _phase_segments(
        selected,
        config.rest_current_threshold_a,
        minimum_sample_count=int(mode_policy["minimum_sample_count"]),
    )
    _add_adjacency(phases)
    records: list[dict[str, Any]] = []

    analysis_scope = _frame_source_interval(selected)
    phase_filter_counts = _pyprobe_phase_counts(
        selected_pyprobe, provider_calls, source_interval=analysis_scope
    )
    for segment in phases:
        flags = list(segment["flags"])
        phase_filter_count = phase_filter_counts.get(segment["phase"])
        if segment["phase"] in {"charge", "discharge", "rest"} and phase_filter_count is None:
            flags.append("pyprobe_phase_filter_unavailable_for_qa")
        records.append(
            make_record(
                record_id=f"operation.phase_segment:{cycle_id}:{segment['segment_index']}",
                record_type="operation.phase_segment",
                cell_id=cell_id,
                cycle_scope=cycle_id,
                source_intervals=[segment["source_interval"]],
                attributes={
                    "phase": segment["phase"],
                    "previous_phase": segment.get("previous_phase"),
                    "next_phase": segment.get("next_phase"),
                    "source_step_index": segment.get("source_step_index"),
                    "segmentation_basis": segment["segmentation_basis"],
                    "phase_label_basis": segment["phase_label_basis"],
                    "pyprobe_window_phase_filter_rows": phase_filter_count,
                    "pyprobe_phase_filter_role": "independent_window_level_QA_only",
                },
                metrics={
                    "duration": metric(segment["duration_s"], "s"),
                    "sample_count": metric(segment["sample_count"], "1"),
                },
                provider="BFL",
                method_name="source_step_and_current_phase_segmentation_v1",
                provider_version="BFL 0.4.0",
                parameters={"rest_current_threshold_a": config.rest_current_threshold_a},
                references=[
                    "https://github.com/shiyunliu-battery/battery-data-standard/blob/main/docs/step-cycle-semantics.md",
                    "https://github.com/battery-data-alliance/battery-data-format",
                    PYPROBE_REFERENCE,
                ],
                quality_status="warning" if flags else "ok",
                quality_flags=flags,
                interpretation_limits=[
                    "Phase labels describe electrical direction, not a named test protocol."
                ],
            )
        )

    for segment in phases:
        mode, mode_attributes, mode_flags = _mode_for_segment(
            selected,
            selected_pyprobe,
            segment,
            phases,
            provider_calls,
            mode_policy,
        )
        segment["mode"] = mode
        records.append(
            make_record(
                record_id=f"operation.mode_segment:{cycle_id}:{segment['segment_index']}",
                record_type="operation.mode_segment",
                cell_id=cell_id,
                cycle_scope=cycle_id,
                source_intervals=[segment["source_interval"]],
                attributes={
                    "mode": mode,
                    "phase": segment["phase"],
                    "source_step_index": segment.get("source_step_index"),
                    "segmentation_basis": segment["segmentation_basis"],
                    **mode_attributes,
                },
                metrics={
                    "duration": metric(segment["duration_s"], "s"),
                    "sample_count": metric(segment["sample_count"], "1"),
                },
                provider="PyProBE+BFL" if selected_pyprobe is not None else "BFL",
                method_name=(
                    "pyprobe_candidates_with_conservative_mode_gate"
                    if selected_pyprobe is not None
                    else "current_shape_mode_gate_v1"
                ),
                provider_version=(
                    f"PyProBE {pyprobe.__version__}; BFL 0.4.0"
                    if selected_pyprobe is not None
                    else "BFL 0.4.0"
                ),
                parameters=mode_policy,
                references=[PYPROBE_REFERENCE],
                applicability_status="unmatched" if mode == "unmatched" else "matched",
                applicability_reasons=mode_flags,
                quality_status="warning" if mode_flags else "ok",
                quality_flags=mode_flags,
                interpretation_limits=[
                    "The closed vocabulary describes waveform shape only.",
                    "A named protocol is emitted only when supplied as declared metadata.",
                ],
            )
        )

    mode_records = [
        record for record in records if record["record_type"] == "operation.mode_segment"
    ]
    records.append(
        _window_summary_record(
            selected,
            phases,
            mode_records,
            config,
            cell_id,
            cycle_id,
            pyprobe_available=selected_pyprobe is not None,
            mode_policy=mode_policy,
        )
    )
    records.append(_exposure_record(selected, config, cell_id, cycle_id))
    return records, phases


def _select_cycle(frame: pl.DataFrame, cycle_id: int | None) -> pl.DataFrame:
    if cycle_id is None or "cycle_index" not in frame.columns:
        return frame.sort("test_time_s")
    return frame.filter(pl.col("cycle_index") == cycle_id).sort("test_time_s")


def _select_pyprobe_cycle(frame: pl.DataFrame | None, cycle_id: int | None) -> pl.DataFrame | None:
    if frame is None or cycle_id is None:
        return frame
    return frame.filter(pl.col("Cycle") == cycle_id)


def _mode_policy_values(config: AnalysisConfig) -> dict[str, Any]:
    """Resolve the exact validated thresholds used by operation mode gates."""

    return {
        "analysis_policy_version": ANALYSIS_POLICY_VERSION,
        "rest_current_threshold_a": config.rest_current_threshold_a,
        "cc_current_cv_limit": config.policy("mode_cc_current_cv_max"),
        "cv_voltage_range_limit_v": config.policy("mode_cv_voltage_range_max_v"),
        "cv_voltage_slope_limit_v_per_s": config.policy("mode_cv_voltage_slope_max_v_per_s"),
        "cv_taper_ratio_limit": config.policy("mode_cv_taper_ratio_max"),
        "cv_nonincreasing_fraction_min": config.policy("mode_cv_nonincreasing_fraction_min"),
        "minimum_sample_count": int(config.policy("mode_min_samples")),
        "cv_taper_noise_tolerance": (
            "max(cv_taper_noise_absolute_floor_a, cv_taper_noise_relative_factor * segment max |I|)"
        ),
        "cv_taper_noise_absolute_floor_a": config.policy("mode_taper_absolute_noise_a"),
        "cv_taper_noise_relative_factor": config.policy("mode_taper_relative_noise_fraction"),
        "pulse_max_duration_s": config.policy("mode_pulse_max_duration_s"),
        "adjacent_rest_min_s": config.policy("mode_adjacent_rest_min_s"),
        "dominant_classified_fraction_min": config.policy("mode_dominant_classified_fraction_min"),
    }


def _phase_segments(
    frame: pl.DataFrame,
    threshold: float,
    *,
    minimum_sample_count: int | None = None,
) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    effective_minimum = (
        minimum_sample_count
        if minimum_sample_count is not None
        else int(DEFAULT_ANALYSIS_POLICY["mode_min_samples"])
    )
    current = frame["current_a"].cast(pl.Float64).to_numpy()
    inferred = np.full(len(current), "unknown", dtype=object)
    finite_current = np.isfinite(current)
    inferred[finite_current & (np.abs(current) <= threshold)] = "rest"
    inferred[finite_current & (current > threshold)] = "charge"
    inferred[finite_current & (current < -threshold)] = "discharge"
    source_phase = _source_phases(frame)
    phases = source_phase if source_phase is not None else inferred
    phase_label_basis = "source_step_type" if source_phase is not None else "current_threshold"
    times = frame["test_time_s"].cast(pl.Float64).to_numpy()
    rows = frame["_source_row"].to_numpy()
    records = frame["record_index"].to_numpy() if "record_index" in frame.columns else rows
    phase_change = phases[1:] != phases[:-1]
    if "step_index" in frame.columns:
        source_steps = (
            frame["step_index"]
            .cast(pl.Int64, strict=False)
            .fill_null(strategy="forward")
            .fill_null(-1)
            .to_numpy()
        )
        boundary_change = phase_change | (source_steps[1:] != source_steps[:-1])
        segmentation_basis = "source_step_or_phase_change"
    else:
        source_steps = None
        boundary_change = phase_change
        segmentation_basis = "phase_change"
    boundaries = np.r_[0, np.flatnonzero(boundary_change) + 1, len(phases)]
    output: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(pairwise(boundaries)):
        end_time = float(times[end - 1])
        if end < len(times):
            end_time = float(times[end])
        duration = max(end_time - float(times[start]), 0.0)
        flags: list[str] = []
        if end - start < effective_minimum:
            flags.append(
                "fewer_than_three_samples"
                if effective_minimum == int(DEFAULT_ANALYSIS_POLICY["mode_min_samples"])
                else "fewer_than_mode_minimum_samples"
            )
        if phases[start] == "unknown":
            flags.append("non_finite_current_phase")
        output.append(
            {
                "segment_index": index,
                "start": int(start),
                "end": int(end),
                "phase": str(phases[start]),
                "source_step_index": (
                    int(source_steps[start]) if source_steps is not None else None
                ),
                "segmentation_basis": segmentation_basis,
                "phase_label_basis": phase_label_basis,
                "duration_s": duration,
                "sample_count": int(end - start),
                "flags": flags,
                "source_interval": {
                    "start_row": int(rows[start]),
                    "end_row": int(rows[end - 1]),
                    "start_record": int(records[start]),
                    "end_record": int(records[end - 1]),
                    "start_time_s": float(times[start]),
                    "end_time_s": end_time,
                    "end_sample_time_s": float(times[end - 1]),
                    "end_time_semantics": "exclusive_previous_zoh_support_boundary",
                },
            }
        )
    return output


def _source_phases(frame: pl.DataFrame) -> np.ndarray | None:
    candidates = [name for name in ("step_type", "raw:Step Type") if name in frame.columns]
    if not candidates:
        return None
    values = frame[candidates[0]].cast(pl.String, strict=False).to_numpy()
    normalized: list[str | None] = []
    for value in values:
        text = "" if value is None else str(value).lower()
        if any(token in text for token in ("discharge", "dchg", "dischg")):
            normalized.append("discharge")
        elif any(token in text for token in ("charge", "chg", "cccv")):
            normalized.append("charge")
        elif any(token in text for token in ("rest", "pause", "ocv", "relax")):
            normalized.append("rest")
        else:
            normalized.append(None)
    if any(value is None for value in normalized):
        return None
    return np.asarray(normalized, dtype=object)


def _add_adjacency(segments: list[dict[str, Any]]) -> None:
    for index, segment in enumerate(segments):
        segment["previous_phase"] = segments[index - 1]["phase"] if index else None
        segment["next_phase"] = segments[index + 1]["phase"] if index + 1 < len(segments) else None


def _pyprobe_phase_counts(
    frame: pl.DataFrame | None,
    calls: list[dict[str, Any]],
    *,
    source_interval: dict[str, Any],
) -> dict[str, int | None]:
    output: dict[str, int | None] = {"charge": None, "discharge": None, "rest": None}
    if frame is None:
        for phase in output:
            calls.append(
                {
                    "provider": "PyProBE",
                    "method": phase,
                    "status": "not_invoked",
                    "reason": "missing_required_channel:voltage_v",
                    "source_interval": source_interval,
                    "role": "window_level_phase_filter_QA",
                }
            )
        return output
    if frame.is_empty():
        return output
    experiment = pyprobe_experiment(frame)
    for phase, _current_count in output.items():  # noqa: PERF102 - mutation during iteration
        try:
            result = getattr(experiment, phase)()
            output[phase] = int(result.data.height)
            calls.append(
                {
                    "provider": "PyProBE",
                    "method": phase,
                    "status": "ok",
                    "rows": output[phase],
                    "source_interval": source_interval,
                    "role": "window_level_phase_filter_QA",
                }
            )
        except Exception as exc:  # noqa: BLE001 - provider errors are audit data
            calls.append(
                {
                    "provider": "PyProBE",
                    "method": phase,
                    "status": "error",
                    "error": str(exc),
                    "source_interval": source_interval,
                    "role": "window_level_phase_filter_QA",
                }
            )
    return output


def _frame_source_interval(frame: pl.DataFrame) -> dict[str, Any]:
    """Return the exact selected canonical scope passed to a window-level provider call."""

    rows = frame["_source_row"].to_numpy()
    records = frame["record_index"].to_numpy() if "record_index" in frame.columns else rows
    times = frame["test_time_s"].cast(pl.Float64).to_numpy()
    return {
        "start_row": int(rows[0]),
        "end_row": int(rows[-1]),
        "start_record": int(records[0]),
        "end_record": int(records[-1]),
        "start_time_s": float(times[0]),
        "end_time_s": float(times[-1]),
    }


def _mode_for_segment(
    canonical: pl.DataFrame,
    py_frame: pl.DataFrame | None,
    segment: dict[str, Any],
    segments: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    mode_policy: dict[str, Any],
) -> tuple[str, dict[str, Any], list[str]]:
    start, end = segment["start"], segment["end"]
    if segment["phase"] in {"rest", "unknown"}:
        return "unmatched", {}, list(segment["flags"])
    if end - start < int(mode_policy["minimum_sample_count"]):
        flags = list(segment["flags"])
        if not {
            "fewer_than_three_samples",
            "fewer_than_mode_minimum_samples",
        }.intersection(flags):
            flags.append("fewer_than_mode_minimum_samples")
        return (
            "unmatched",
            {},
            flags,
        )
    sub = canonical.slice(start, end - start)
    current = sub["current_a"].cast(pl.Float64).to_numpy()
    time = sub["test_time_s"].cast(pl.Float64).to_numpy()
    mean_abs = float(np.nanmean(np.abs(current)))
    cv = float(np.nanstd(current) / max(abs(float(np.nanmean(current))), 1e-12))
    has_voltage = "voltage_v" in sub.columns
    if has_voltage:
        voltage = sub["voltage_v"].cast(pl.Float64, strict=False).to_numpy()
        finite_voltage = voltage[np.isfinite(voltage)]
        voltage_range = (
            float(np.max(finite_voltage) - np.min(finite_voltage)) if len(finite_voltage) else None
        )
        duration = max(float(time[-1] - time[0]), 1e-12)
        voltage_slope = (
            abs(float(voltage[-1] - voltage[0]) / duration)
            if np.isfinite(voltage[0]) and np.isfinite(voltage[-1])
            else None
        )
    else:
        voltage_range = None
        voltage_slope = None
    abs_current = np.abs(current[np.isfinite(current)])
    taper_ratio = (
        float(abs_current[-1] / max(abs_current[0], 1e-12)) if len(abs_current) else np.inf
    )
    taper_noise_tolerance = (
        max(
            float(mode_policy["cv_taper_noise_absolute_floor_a"]),
            float(mode_policy["cv_taper_noise_relative_factor"]) * float(np.max(abs_current)),
        )
        if len(abs_current)
        else float(mode_policy["cv_taper_noise_absolute_floor_a"])
    )
    nonincreasing_fraction = (
        float(np.mean(np.diff(abs_current) <= taper_noise_tolerance))
        if len(abs_current) > 1
        else 0.0
    )

    candidate_counts: dict[str, int | None] = {
        "constant_current": None if py_frame is None else 0,
        "constant_voltage": None if py_frame is None else 0,
    }
    provider_error: str | None = None
    if py_frame is None:
        for name in candidate_counts:
            calls.append(
                {
                    "provider": "PyProBE",
                    "method": name,
                    "status": "not_invoked",
                    "reason": "missing_required_channel:voltage_v",
                    "source_interval": segment["source_interval"],
                }
            )
    else:
        py_sub = py_frame.slice(start, end - start)
        try:
            exp = pyprobe_experiment(py_sub)
            for name, _current_count in candidate_counts.items():  # noqa: PERF102 - mutation during iteration
                candidate_counts[name] = int(getattr(exp, name)().data.height)
                calls.append(
                    {
                        "provider": "PyProBE",
                        "method": name,
                        "status": "ok",
                        "rows": candidate_counts[name],
                        "source_interval": segment["source_interval"],
                    }
                )
        except Exception as exc:  # noqa: BLE001 - provider errors are audit data
            provider_error = str(exc)
            calls.append(
                {
                    "provider": "PyProBE",
                    "method": "constant_current/constant_voltage",
                    "status": "error",
                    "error": str(exc),
                    "source_interval": segment["source_interval"],
                }
            )

    if provider_error is not None:
        return (
            "unmatched",
            {"pyprobe_candidate_rows": candidate_counts},
            ["provider_error"],
        )

    index = int(segment["segment_index"])
    adjacent_rests = [
        item
        for item in (
            segments[index - 1] if index else None,
            segments[index + 1] if index + 1 < len(segments) else None,
        )
        if item is not None and item["phase"] == "rest"
    ]
    rest_duration = max((item["duration_s"] for item in adjacent_rests), default=0.0)
    if (
        segment["duration_s"] <= float(mode_policy["pulse_max_duration_s"])
        and rest_duration >= float(mode_policy["adjacent_rest_min_s"])
        and rest_duration >= segment["duration_s"]
    ):
        mode = "pulse_like"
    elif (
        py_frame is not None
        and candidate_counts["constant_voltage"] is not None
        and candidate_counts["constant_voltage"] > 0
        and voltage_range is not None
        and voltage_range <= float(mode_policy["cv_voltage_range_limit_v"])
        and voltage_slope is not None
        and voltage_slope <= float(mode_policy["cv_voltage_slope_limit_v_per_s"])
        and taper_ratio <= float(mode_policy["cv_taper_ratio_limit"])
        and nonincreasing_fraction >= float(mode_policy["cv_nonincreasing_fraction_min"])
    ):
        mode = "constant_voltage_like"
    elif (
        (
            py_frame is None
            or (
                candidate_counts["constant_current"] is not None
                and candidate_counts["constant_current"] > 0
            )
        )
        and mean_abs > float(mode_policy["rest_current_threshold_a"])
        and cv <= float(mode_policy["cc_current_cv_limit"])
        and np.all(np.sign(current[np.isfinite(current)]) == np.sign(np.nanmean(current)))
    ):
        mode = "constant_current_like"
    else:
        mode = "dynamic_current"
    return (
        mode,
        {
            "pyprobe_candidate_rows": candidate_counts,
            "mode_provider_branch": (
                "pyprobe_candidates_with_bfl_gates"
                if py_frame is not None
                else "bfl_current_shape_only"
            ),
            "current_coefficient_of_variation": json_number(cv),
            "voltage_range_v": json_number(voltage_range),
            "voltage_slope_v_per_s": json_number(voltage_slope),
            "current_taper_ratio": json_number(taper_ratio),
            "nonincreasing_current_fraction": json_number(nonincreasing_fraction),
            "taper_noise_tolerance_a": json_number(taper_noise_tolerance),
        },
        [],
    )


def _window_summary_record(
    frame: pl.DataFrame,
    phases: list[dict[str, Any]],
    mode_records: list[dict[str, Any]],
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
    *,
    pyprobe_available: bool,
    mode_policy: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate thousands of segments into one LLM-ready operating fingerprint."""

    phase_names = ("charge", "discharge", "rest", "unknown")
    mode_names = (
        "constant_current_like",
        "constant_voltage_like",
        "pulse_like",
        "dynamic_current",
        "unmatched",
    )
    phase_duration = {
        name: float(sum(item["duration_s"] for item in phases if item["phase"] == name))
        for name in phase_names
    }
    phase_count = {name: sum(item["phase"] == name for item in phases) for name in phase_names}
    mode_duration = {
        name: float(
            sum(
                record["metrics"]["duration"]["value"] or 0.0
                for record in mode_records
                if record["attributes"]["mode"] == name
            )
        )
        for name in mode_names
    }
    mode_count = {
        name: sum(record["attributes"]["mode"] == name for record in mode_records)
        for name in mode_names
    }
    total_duration = float(sum(phase_duration.values()))
    active_duration = phase_duration["charge"] + phase_duration["discharge"]
    active_mode_duration = {
        name: float(
            sum(
                record["metrics"]["duration"]["value"] or 0.0
                for record in mode_records
                if record["attributes"]["mode"] == name
                and record["attributes"]["phase"] in {"charge", "discharge"}
            )
        )
        for name in mode_names
    }
    comparable_modes = [
        name for name in mode_names if name != "unmatched" and active_mode_duration[name] > 0
    ]
    classified_active_duration = sum(
        active_mode_duration[name] for name in mode_names if name != "unmatched"
    )
    classified_active_fraction = (
        classified_active_duration / active_duration if active_duration else None
    )
    dominant_candidate = (
        max(comparable_modes, key=active_mode_duration.__getitem__) if comparable_modes else None
    )
    dominant_mode = (
        dominant_candidate
        if classified_active_fraction is not None
        and classified_active_fraction >= float(mode_policy["dominant_classified_fraction_min"])
        else None
    )
    active_durations = [item["duration_s"] for item in phases if item["phase"] != "rest"]
    rows = frame["_source_row"].to_numpy()
    time = frame["test_time_s"].cast(pl.Float64).to_numpy()
    current = frame["current_a"].cast(pl.Float64, strict=False).to_numpy()
    dt = np.diff(time)
    current_interval_valid = np.isfinite(dt) & (dt > 0) & np.isfinite(current[:-1])
    charge_mask = current_interval_valid & (current[:-1] > config.rest_current_threshold_a)
    discharge_mask = current_interval_valid & (current[:-1] < -config.rest_current_threshold_a)
    charge_throughput = float(np.sum(current[:-1][charge_mask] * dt[charge_mask]) / 3600.0)
    discharge_throughput = float(
        -np.sum(current[:-1][discharge_mask] * dt[discharge_mask]) / 3600.0
    )
    current_intervals = previous_intervals(
        time,
        np.abs(current),
        sampling_interval_outlier_factor=config.sampling_interval_outlier_factor,
    )
    current_summary = weighted_summary(current_intervals)
    has_voltage = "voltage_v" in frame.columns
    if has_voltage:
        voltage = frame["voltage_v"].cast(pl.Float64, strict=False).to_numpy()
        voltage_interval_valid = current_interval_valid & np.isfinite(voltage[:-1])
        energy_throughput = (
            float(
                np.sum(
                    np.abs(
                        current[:-1][voltage_interval_valid] * voltage[:-1][voltage_interval_valid]
                    )
                    * dt[voltage_interval_valid]
                )
                / 3600.0
            )
            if np.any(voltage_interval_valid)
            else None
        )
        voltage_intervals = previous_intervals(
            time,
            voltage,
            sampling_interval_outlier_factor=config.sampling_interval_outlier_factor,
        )
        voltage_summary = weighted_summary(voltage_intervals)
        power_intervals = previous_intervals(
            time,
            np.abs(current * voltage),
            sampling_interval_outlier_factor=config.sampling_interval_outlier_factor,
        )
        power_summary = weighted_summary(power_intervals)
        voltage_reason = (
            None
            if voltage_intervals.included_duration_s > 0
            else "voltage_channel_has_no_usable_intervals"
        )
    else:
        energy_throughput = None
        voltage_summary = {
            name: None for name in ("min", "max", "mean", "rms", "q05", "q50", "q95", "q99")
        }
        power_summary = dict(voltage_summary)
        voltage_reason = "missing_required_channel:voltage_v"
    current_squared_exposure = float(
        np.sum(np.square(current[:-1][current_interval_valid]) * dt[current_interval_valid])
        / 3600.0
    )
    temp_name = temperature_column(frame)
    if temp_name is not None:
        temperature = frame[temp_name].cast(pl.Float64, strict=False).to_numpy()
        temperature_intervals = previous_intervals(
            time,
            temperature,
            sampling_interval_outlier_factor=config.sampling_interval_outlier_factor,
        )
        temperature_summary = weighted_summary(temperature_intervals)
        temperature_coverage = (
            temperature_intervals.included_duration_s / temperature_intervals.total_duration_s
            if temperature_intervals.total_duration_s
            else None
        )
        temperature_reason = (
            None
            if temperature_intervals.included_duration_s > 0
            else "temperature_channel_has_no_usable_intervals"
        )
    else:
        temperature_summary = {
            name: None for name in ("min", "max", "mean", "rms", "q05", "q50", "q95", "q99")
        }
        temperature_coverage = None
        temperature_reason = "missing_required_channel:temperature"
    source_records = frame["record_index"].to_numpy() if "record_index" in frame.columns else rows
    flags = sorted({flag for record in mode_records for flag in record["quality"]["flags"]})
    sequence_limit = 32
    operation_sequence = [
        {
            "segment_index": int(phase["segment_index"]),
            "source_step_index": phase.get("source_step_index"),
            "phase": phase["phase"],
            "mode": mode_record["attributes"]["mode"],
            "duration_s": phase["duration_s"],
            "phase_record_id": f"operation.phase_segment:{cycle_id}:{phase['segment_index']}",
            "mode_record_id": f"operation.mode_segment:{cycle_id}:{phase['segment_index']}",
        }
        for phase, mode_record in zip(phases, mode_records, strict=True)
    ]
    metrics = {
        "total_duration": metric(total_duration, "s"),
        "active_duration": metric(active_duration, "s"),
        "rest_fraction": metric(
            phase_duration["rest"] / total_duration if total_duration else None,
            "1",
        ),
        "charge_fraction": metric(
            phase_duration["charge"] / total_duration if total_duration else None,
            "1",
        ),
        "discharge_fraction": metric(
            phase_duration["discharge"] / total_duration if total_duration else None,
            "1",
        ),
        "unknown_fraction": metric(
            phase_duration["unknown"] / total_duration if total_duration else None,
            "1",
        ),
        "phase_transition_count": metric(max(len(phases) - 1, 0), "1"),
        "phase_transitions_per_hour": metric(
            max(len(phases) - 1, 0) / (total_duration / 3600.0) if total_duration else None,
            "1/h",
        ),
        "active_segment_duration_q50": metric(
            json_number(np.quantile(active_durations, 0.5)) if active_durations else None,
            "s",
        ),
        "active_segment_duration_q95": metric(
            json_number(np.quantile(active_durations, 0.95)) if active_durations else None,
            "s",
        ),
        "classified_active_fraction": metric(classified_active_fraction, "1"),
        "dominant_mode_active_fraction": metric(
            active_mode_duration[dominant_mode] / active_duration
            if dominant_mode is not None and active_duration
            else None,
            "1",
        ),
        "charge_throughput": metric(charge_throughput, "Ah"),
        "discharge_throughput": metric(discharge_throughput, "Ah"),
        "capacity_throughput": metric(charge_throughput + discharge_throughput, "Ah"),
        "energy_throughput": metric(energy_throughput, "Wh", reason=voltage_reason),
        "absolute_current_q50": metric(current_summary["q50"], "A"),
        "absolute_current_q95": metric(current_summary["q95"], "A"),
        "absolute_power_q95": metric(power_summary["q95"], "W", reason=voltage_reason),
        "current_squared_exposure": metric(current_squared_exposure, "A^2 h"),
        "voltage_min": metric(voltage_summary["min"], "V", reason=voltage_reason),
        "voltage_q05": metric(voltage_summary["q05"], "V", reason=voltage_reason),
        "voltage_q50": metric(voltage_summary["q50"], "V", reason=voltage_reason),
        "voltage_q95": metric(voltage_summary["q95"], "V", reason=voltage_reason),
        "voltage_max": metric(voltage_summary["max"], "V", reason=voltage_reason),
        "temperature_min": metric(
            temperature_summary["min"],
            "degC",
            reason=temperature_reason,
        ),
        "temperature_q50": metric(
            temperature_summary["q50"],
            "degC",
            reason=temperature_reason,
        ),
        "temperature_q95": metric(
            temperature_summary["q95"],
            "degC",
            reason=temperature_reason,
        ),
        "temperature_max": metric(
            temperature_summary["max"],
            "degC",
            reason=temperature_reason,
        ),
        "temperature_duration_coverage": metric(
            temperature_coverage,
            "1",
            reason=temperature_reason,
        ),
    }
    for name in mode_names:
        metrics[f"{name}_active_fraction"] = metric(
            active_mode_duration[name] / active_duration if active_duration else None,
            "1",
        )
    return make_record(
        record_id=f"operation.window_summary:{cycle_id}",
        record_type="operation.window_summary",
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
            "phase_segment_counts": phase_count,
            "phase_durations_s": phase_duration,
            "mode_segment_counts": mode_count,
            "mode_durations_s": mode_duration,
            "active_mode_durations_s": active_mode_duration,
            "operation_sequence": operation_sequence if len(phases) <= sequence_limit else [],
            "operation_sequence_status": (
                "included" if len(phases) <= sequence_limit else "retrieve_segment_records"
            ),
            "operation_sequence_summary_limit": sequence_limit,
            "dominant_active_mode": dominant_mode,
            "dominant_active_mode_candidate": dominant_candidate,
            "dominant_mode_minimum_classified_active_fraction": (
                mode_policy["dominant_classified_fraction_min"]
            ),
            "reference_frame": {
                "current_sign": "charge-positive",
                "scope": "selected_representative_cycle"
                if cycle_id is not None
                else "complete_observation_window",
                "capacity_reference_ah": config.nominal_capacity_ah,
                "temperature_column": temp_name,
            },
            "confidence": {
                "level": "medium" if flags else "high",
                "basis": ["source_grounded_phase_segments", "duration_weighting", *flags],
                "not_a_probability": True,
            },
        },
        metrics=metrics,
        provider="PyProBE+BFL" if pyprobe_available else "BFL",
        method_name="source_grounded_operating_window_v1",
        provider_version=(
            f"PyProBE {pyprobe.__version__}; BFL 0.4.0" if pyprobe_available else "BFL 0.4.0"
        ),
        parameters={
            "rest_current_threshold_a": config.rest_current_threshold_a,
            "duration_weighting": True,
            "interval_semantics": "x[i] holds on [t[i], t[i+1])",
            "mode_policy": mode_policy,
            "dominant_mode_minimum_classified_active_fraction": (
                mode_policy["dominant_classified_fraction_min"]
            ),
        },
        references=[
            PYPROBE_REFERENCE,
            "https://doi.org/10.1038/s41597-024-03831-x",
            "https://rovi-org.github.io/battery-data-toolkit/user-guide/post-processing/index.html",
        ],
        quality_status="warning" if flags else "ok",
        quality_flags=flags,
        interpretation_limits=[
            "The profile is a waveform fingerprint, not an inferred named protocol.",
            "Duration fractions describe the selected source scope and do not imply aging severity.",
        ],
    )


def _exposure_record(
    frame: pl.DataFrame,
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
) -> dict[str, Any]:
    time = frame["test_time_s"].cast(pl.Float64).to_numpy()
    current = frame["current_a"].cast(pl.Float64).to_numpy()
    intervals = previous_intervals(
        time,
        current,
        sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
    )
    summary = weighted_summary(intervals)
    absolute_intervals = previous_intervals(
        time,
        np.abs(current),
        sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
    )
    absolute_summary = weighted_summary(absolute_intervals)
    throughput, _ = integrate_previous(
        time,
        np.abs(current),
        scale=1.0 / 3600.0,
        sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
    )
    valid = intervals.valid
    duration = float(np.sum(intervals.durations_s[valid])) if np.any(valid) else 0.0
    charge_duration = float(
        np.sum(intervals.durations_s[valid & (intervals.values > config.rest_current_threshold_a)])
    )
    discharge_duration = float(
        np.sum(intervals.durations_s[valid & (intervals.values < -config.rest_current_threshold_a)])
    )
    rest_duration = max(duration - charge_duration - discharge_duration, 0.0)
    flags: list[str] = []
    if intervals.excluded_duration_s:
        flags.append("excluded_non_finite_duration")
    if intervals.non_positive_interval_count:
        flags.append("non_positive_time_intervals")
    if intervals.sampling_interval_outlier_count:
        flags.append("sampling_interval_outlier")

    absolute_q99 = weighted_quantile(np.abs(intervals.values), intervals.durations_s, [0.99])[0]
    isolated_max = False
    if np.isfinite(absolute_q99):
        indexes = np.flatnonzero(np.abs(current) >= absolute_q99)
        runs = np.split(indexes, np.where(np.diff(indexes) > 1)[0] + 1)
        isolated_max = any(
            len(run) == 1 and np.isclose(abs(current[run[0]]), np.nanmax(np.abs(current)))
            for run in runs
            if len(run)
        )

    metrics = {name: metric(value, "A") for name, value in summary.items()}
    metrics.update(
        {
            **{
                f"absolute_current_{name}": metric(value, "A")
                for name, value in absolute_summary.items()
            },
            "capacity_throughput": metric(throughput, "Ah"),
            "included_duration": metric(intervals.included_duration_s, "s"),
            "excluded_duration": metric(intervals.excluded_duration_s, "s"),
            "sampling_interval_outlier_count": metric(
                intervals.sampling_interval_outlier_count, "1"
            ),
            "sampling_interval_outlier_duration": metric(
                intervals.sampling_interval_outlier_duration_s, "s"
            ),
            "maximum_sampling_interval_outlier": metric(
                intervals.max_sampling_interval_outlier_s, "s"
            ),
            "charge_duration": metric(charge_duration, "s"),
            "discharge_duration": metric(discharge_duration, "s"),
            "rest_duration": metric(rest_duration, "s"),
            "charge_fraction": metric(charge_duration / duration if duration else None, "1"),
            "discharge_fraction": metric(discharge_duration / duration if duration else None, "1"),
            "rest_fraction": metric(rest_duration / duration if duration else None, "1"),
        }
    )
    if config.nominal_capacity_ah:
        for name in ("mean", "rms", "q05", "q50", "q95", "q99"):
            metrics[f"c_rate_{name}"] = metric(
                float(absolute_summary[name]) / config.nominal_capacity_ah
                if absolute_summary[name] is not None
                else None,
                "1/h",
            )
    rows = frame["_source_row"].to_numpy()
    source_records = frame["record_index"].to_numpy() if "record_index" in frame.columns else rows
    return make_record(
        record_id=f"operation.exposure_summary:{cycle_id}",
        record_type="operation.exposure_summary",
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
        attributes={"isolated_single_point_maximum": isolated_max},
        metrics=metrics,
        provider="BFL",
        method_name="duration_weighted_previous_zoh_v1",
        provider_version="0.4.0",
        parameters={
            "interval_semantics": "x[i] holds on [t[i], t[i+1])",
            "sampling_interval_outlier_factor": (config.sampling_interval_outlier_factor),
            "sampling_interval_outlier_detection": (
                "isolated_interval_vs_nearest_positive_neighbors_v1"
            ),
            "sampling_interval_outliers_included_in_integrals": True,
        },
        references=["https://numpy.org/doc/stable/reference/generated/numpy.histogram.html"],
        quality_status="warning" if flags else "ok",
        quality_flags=flags,
        interpretation_limits=["Exposure statistics describe only the selected source scope."],
    )
