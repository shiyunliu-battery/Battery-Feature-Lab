"""Cycle response, rest/thermal, pulse resistance, and ICA/DVA records."""

from __future__ import annotations

from collections import Counter
from typing import Any

import bds
import numpy as np
import polars as pl
import pyprobe
import scipy
from pyprobe.analysis import cycling, differentiation, pulsing
from scipy.signal import find_peaks, peak_prominences

from battery_feature_lab.analysis.adapters import (
    analysis_capacity,
    pyprobe_experiment,
    pyprobe_result,
    temperature_column,
)
from battery_feature_lab.analysis.numerics import (
    finite_coverage,
    integrate_previous,
    json_number,
    mostly_monotone,
    previous_intervals,
    reported_delta,
)
from battery_feature_lab.analysis.schema import (
    ANALYSIS_POLICY_VERSION,
    AnalysisConfig,
    make_record,
    metric,
    seconds_label,
)

PYPROBE_REFERENCE = "https://doi.org/10.21105/joss.07474"
LEAN_REFERENCE = "https://doi.org/10.1016/j.etran.2020.100051"


def analyze_cycles(
    frame: pl.DataFrame,
    pyprobe_frame: pl.DataFrame,
    *,
    config: AnalysisConfig,
    cell_id: str,
    cycle_id_source: str,
    provider_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create one cycle summary record per available cycle."""

    if "cycle_index" not in frame.columns:
        return [_cycle_unavailable_record(cell_id)], []
    cycle_ids = [
        int(value) for value in frame["cycle_index"].drop_nulls().unique(maintain_order=True)
    ]
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    first_cycle_id = cycle_ids[0] if cycle_ids else None
    last_cycle_id = cycle_ids[-1] if cycle_ids else None
    for cycle_id in cycle_ids:
        cycle_frame = frame.filter(pl.col("cycle_index") == cycle_id).sort("test_time_s")
        cycle_py = pyprobe_frame.filter(pl.col("Cycle") == cycle_id).sort("Time [s]")
        cycle_capacity, cycle_adapter_details = analysis_capacity(
            cycle_frame,
            sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
            reported_capacity_finite_coverage_min=config.policy(
                "reported_capacity_finite_coverage_min"
            ),
        )
        cycle_py = cycle_py.with_columns(pl.Series("Capacity [Ah]", cycle_capacity))
        record, summary = _cycle_record(
            cycle_frame,
            cycle_py,
            cycle_id=cycle_id,
            cycle_id_source=cycle_id_source,
            adapter_details=cycle_adapter_details,
            touches_dataset_start=cycle_id == first_cycle_id,
            touches_dataset_end=cycle_id == last_cycle_id,
            config=config,
            cell_id=cell_id,
            provider_calls=provider_calls,
        )
        records.append(record)
        summaries.append(summary)
    return records, summaries


def analyze_rest_and_thermal(
    frame: pl.DataFrame,
    pyprobe_frame: pl.DataFrame,
    phases: list[dict[str, Any]],
    *,
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
    provider_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create voltage relaxation and same-interval thermal descriptors."""

    selected = _select_cycle(frame, cycle_id)
    selected_py = _select_py_cycle(pyprobe_frame, cycle_id)
    selected_interval = _frame_source_interval(selected)
    try:
        pyprobe_rest = pyprobe_experiment(selected_py).rest().data
        if "_source_row" not in pyprobe_rest.columns:
            raise ValueError("PyProBE rest result did not preserve _source_row")
        confirmed_rest_rows = set(pyprobe_rest["_source_row"].cast(pl.Int64).to_list())
        provider_calls.append(
            {
                "provider": "PyProBE",
                "method": "rest",
                "status": "ok",
                "rows": pyprobe_rest.height,
                "source_interval": selected_interval,
            }
        )
    except Exception as exc:  # noqa: BLE001 - provider errors are audit data
        provider_calls.append(
            {
                "provider": "PyProBE",
                "method": "rest",
                "status": "error",
                "error": str(exc),
                "source_interval": selected_interval,
            }
        )
        return [
            _provider_unavailable_record(
                record_id=f"response.rest_and_thermal:{cycle_id}:unavailable",
                record_type="response.rest_and_thermal",
                cell_id=cell_id,
                cycle_id=cycle_id,
                provider="PyProBE+BFL",
                method="rest_filter_with_interval_descriptors_v1",
                reason=f"provider_error: {exc}",
                version=f"PyProBE {pyprobe.__version__}; BFL 0.4.0",
            )
        ]

    records: list[dict[str, Any]] = []
    temp_column = temperature_column(selected)
    for segment in phases:
        if segment["phase"] != "rest":
            continue
        immediate_previous = (
            phases[segment["segment_index"] - 1] if segment["segment_index"] > 0 else None
        )
        previous_segment = (
            immediate_previous
            if immediate_previous and immediate_previous["phase"] in {"charge", "discharge"}
            else None
        )
        sub = selected.slice(segment["start"], segment["end"] - segment["start"])
        segment_rows = set(sub["_source_row"].cast(pl.Int64).to_list())
        if not segment_rows or not segment_rows.issubset(confirmed_rest_rows):
            records.append(
                _provider_unavailable_record(
                    record_id=f"response.rest_and_thermal:{cycle_id}:{segment['segment_index']}",
                    record_type="response.rest_and_thermal",
                    cell_id=cell_id,
                    cycle_id=cycle_id,
                    provider="PyProBE+BFL",
                    method="rest_filter_with_interval_descriptors_v1",
                    reason="pyprobe_rest_filter_did_not_confirm_complete_interval",
                    version=f"PyProBE {pyprobe.__version__}; BFL 0.4.0",
                    source_interval=segment["source_interval"],
                )
            )
            continue
        time = sub["test_time_s"].cast(pl.Float64).to_numpy()
        voltage = sub["voltage_v"].cast(pl.Float64).to_numpy()
        finite_voltage = np.isfinite(time) & np.isfinite(voltage)
        finite_time = time[finite_voltage]
        finite_values = voltage[finite_voltage]
        relative = finite_time - finite_time[0] if len(finite_time) else np.asarray([], dtype=float)
        observed_sample_span = float(relative[-1]) if len(relative) else 0.0
        support_duration = float(segment["duration_s"])
        intervals = previous_intervals(
            finite_time,
            finite_values,
            sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
        )
        enough_voltage = len(finite_values) >= 2
        metrics = {
            "duration": metric(support_duration, "s"),
            "observed_sample_span": metric(observed_sample_span, "s"),
            "finite_voltage_sample_count": metric(len(finite_values), "1"),
            "voltage_sample_coverage": metric(
                len(finite_values) / len(voltage) if len(voltage) else None,
                "1",
            ),
            "start_voltage": metric(
                json_number(finite_values[0]) if len(finite_values) else None,
                "V",
                status="ok" if len(finite_values) else "not_computable",
                reason=None if len(finite_values) else "no_finite_rest_voltage",
            ),
            "end_voltage": metric(
                json_number(finite_values[-1]) if len(finite_values) else None,
                "V",
                status="ok" if len(finite_values) else "not_computable",
                reason=None if len(finite_values) else "no_finite_rest_voltage",
            ),
            "voltage_change": metric(
                json_number(finite_values[-1] - finite_values[0]) if enough_voltage else None,
                "V",
                status="ok" if enough_voltage else "not_computable",
                reason=(None if enough_voltage else "fewer_than_two_finite_rest_voltage_samples"),
            ),
            "preceding_load_duration": metric(
                previous_segment["duration_s"] if previous_segment else None,
                "s",
            ),
        }
        if previous_segment is not None:
            previous_sub = selected.slice(
                previous_segment["start"],
                previous_segment["end"] - previous_segment["start"],
            )
            previous_current = previous_sub["current_a"].cast(pl.Float64).to_numpy()
            finite_previous_current = previous_current[np.isfinite(previous_current)]
        else:
            finite_previous_current = np.asarray([], dtype=float)
        metrics["preceding_load_median_current"] = metric(
            json_number(np.median(finite_previous_current))
            if len(finite_previous_current)
            else None,
            "A",
        )
        metrics["preceding_load_end_current"] = metric(
            json_number(finite_previous_current[-1]) if len(finite_previous_current) else None,
            "A",
        )
        for checkpoint in config.relaxation_checkpoints_s:
            checkpoint = float(checkpoint)
            checkpoint_label = seconds_label(checkpoint)
            value, bracket_width, nearest_offset, checkpoint_reason = _interpolate_checkpoint(
                relative,
                finite_values,
                checkpoint,
                sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
            )
            available = value is not None
            metrics[f"voltage_at_{checkpoint_label}s"] = metric(
                value,
                "V",
                status="ok" if available else "not_computable",
                reason=checkpoint_reason,
            )
            metrics[f"voltage_change_at_{checkpoint_label}s"] = metric(
                value - float(finite_values[0]) if value is not None else None,
                "V",
                status="ok" if available else "not_computable",
                reason=checkpoint_reason,
            )
            metrics[f"checkpoint_bracket_width_{checkpoint_label}s"] = metric(
                bracket_width,
                "s",
                status="ok" if bracket_width is not None else "not_computable",
                reason=checkpoint_reason,
            )
            metrics[f"checkpoint_nearest_sample_offset_{checkpoint_label}s"] = metric(
                nearest_offset,
                "s",
                status="ok" if nearest_offset is not None else "not_computable",
                reason=checkpoint_reason,
            )
        if len(finite_values) >= 10:
            total_change = float(finite_values[-1] - finite_values[0])
            differences = np.diff(finite_values)
            direction = np.sign(total_change)
            tolerance = max(1e-6, 0.001 * float(np.ptp(finite_values)))
            monotone_fraction = (
                float(np.mean(direction * differences >= -tolerance)) if direction != 0 else None
            )
            positive_time = relative > 0
            if int(np.sum(positive_time)) >= 10:
                log_slope = float(
                    np.polyfit(
                        np.log10(relative[positive_time]),
                        finite_values[positive_time],
                        1,
                    )[0]
                )
            else:
                log_slope = None
        else:
            monotone_fraction = None
            log_slope = None
        metrics["voltage_relaxation_monotone_fraction"] = metric(
            monotone_fraction,
            "1",
            reason="fewer_than_10_finite_rest_samples" if monotone_fraction is None else None,
        )
        metrics["voltage_vs_log10_time_slope"] = metric(
            log_slope,
            "V/decade",
            reason="fewer_than_10_positive_time_samples" if log_slope is None else None,
        )
        change_60 = metrics.get("voltage_change_at_60s", {}).get("value")
        change_1800 = metrics.get("voltage_change_at_1800s", {}).get("value")
        shape_ratio = (
            abs(float(change_60)) / abs(float(change_1800))
            if change_60 is not None and change_1800 is not None and abs(float(change_1800)) > 1e-12
            else None
        )
        metrics["absolute_change_60s_fraction_of_1800s"] = metric(
            shape_ratio,
            "1",
            reason="60s_and_1800s_changes_not_both_resolved" if shape_ratio is None else None,
        )
        flags: list[str] = []
        if len(finite_values) < len(voltage):
            flags.append("non_finite_rest_voltage_samples_excluded")
        if intervals.non_positive_interval_count:
            flags.append("non_positive_rest_time_intervals")
        if intervals.sampling_interval_outlier_count:
            flags.append("sampling_interval_outlier")
        if temp_column is None:
            for name in (
                "temperature_baseline",
                "temperature_max",
                "temperature_rise",
                "time_to_temperature_max",
            ):
                metrics[name] = metric(
                    None,
                    "s" if name.startswith("time_") else "degC",
                    status="not_computable",
                    reason="no standardized temperature channel",
                )
            flags.append("temperature_unavailable")
        else:
            temperature = sub[temp_column].cast(pl.Float64, strict=False).to_numpy()
            finite = np.flatnonzero(np.isfinite(temperature))
            if len(finite):
                baseline = float(temperature[finite[0]])
                maximum_index = int(finite[np.argmax(temperature[finite])])
                maximum = float(temperature[maximum_index])
                metrics.update(
                    {
                        "temperature_baseline": metric(baseline, "degC"),
                        "temperature_max": metric(maximum, "degC"),
                        "temperature_rise": metric(maximum - baseline, "degC"),
                        "time_to_temperature_max": metric(
                            float(time[maximum_index] - time[0]), "s"
                        ),
                    }
                )
            else:
                flags.append("temperature_all_missing")
                for name in (
                    "temperature_baseline",
                    "temperature_max",
                    "temperature_rise",
                    "time_to_temperature_max",
                ):
                    metrics[name] = metric(
                        None,
                        "s" if name.startswith("time_") else "degC",
                        status="not_computable",
                        reason="temperature values are missing",
                    )

        records.append(
            make_record(
                record_id=f"response.rest_and_thermal:{cycle_id}:{segment['segment_index']}",
                record_type="response.rest_and_thermal",
                cell_id=cell_id,
                cycle_scope=cycle_id,
                source_intervals=[segment["source_interval"]],
                attributes={
                    "temperature_column": temp_column,
                    "previous_phase": segment.get("previous_phase"),
                    "next_phase": segment.get("next_phase"),
                    "pyprobe_rest_filter_confirmed": True,
                    "preceding_segment_id": (
                        f"operation.phase_segment:{cycle_id}:{previous_segment['segment_index']}"
                        if previous_segment
                        else None
                    ),
                    "preceding_mode": (previous_segment.get("mode") if previous_segment else None),
                },
                metrics=metrics,
                provider="PyProBE+BFL",
                method_name="rest_filter_with_interval_descriptors_v1",
                provider_version=f"PyProBE {pyprobe.__version__}; BFL 0.4.0",
                parameters={
                    "analysis_policy_version": ANALYSIS_POLICY_VERSION,
                    "checkpoints_s": list(config.relaxation_checkpoints_s),
                    "checkpoint_method": "linear_interpolation_within_observed_rest",
                    "shape_method": "monotone_fraction_and_voltage_vs_log10_time_slope",
                    "duration_semantics": "previous_zoh_support_to_next_phase_boundary",
                    "observed_sample_span_semantics": (
                        "last_finite_sample_time_minus_first_finite_sample_time"
                    ),
                    "sampling_interval_outlier_factor": (config.sampling_interval_outlier_factor),
                },
                references=[PYPROBE_REFERENCE],
                quality_status="warning" if flags else "ok",
                quality_flags=flags,
                applicability_status=("applicable" if enough_voltage else "not_computable"),
                applicability_reasons=(
                    [] if enough_voltage else ["fewer_than_two_finite_rest_voltage_samples"]
                ),
                interpretation_limits=[
                    "Voltage change is a terminal response, not equilibrium voltage.",
                    "Temperature co-occurrence does not establish causality.",
                ],
            )
        )
    if not records:
        records.append(
            _provider_unavailable_record(
                record_id=f"response.rest_and_thermal:{cycle_id}:unavailable",
                record_type="response.rest_and_thermal",
                cell_id=cell_id,
                cycle_id=cycle_id,
                provider="PyProBE+BFL",
                method="rest_filter_with_interval_descriptors_v1",
                reason="no_rest_interval_in_representative_scope",
                version=f"PyProBE {pyprobe.__version__}; BFL 0.4.0",
            )
        )
    return records


def analyze_pulses(
    frame: pl.DataFrame,
    pyprobe_frame: pl.DataFrame,
    phases: list[dict[str, Any]],
    cycle_summaries: list[dict[str, Any]],
    *,
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
    provider_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use PyProBE for eligible pulse resistance records."""

    selected_py = _select_py_cycle(pyprobe_frame, cycle_id)
    reference_capacity = config.nominal_capacity_ah or _reference_capacity(cycle_summaries)
    soc_anchor = _soc_anchor(cycle_summaries, cycle_id)
    candidates = _pulse_candidates(
        phases,
        max(config.pulse_resistance_times_s),
        rest_min_s=config.policy("current_step_rest_min_s"),
        pulse_max_s=config.policy("current_step_pulse_max_s"),
    )
    if not candidates:
        return [
            _provider_unavailable_record(
                record_id=f"response.pulse_resistance:{cycle_id}:unavailable",
                record_type="response.pulse_resistance",
                cell_id=cell_id,
                cycle_id=cycle_id,
                provider="PyProBE",
                method="pulsing.get_resistances",
                reason="no_interval_passed_pulse_rest_gate",
                version=pyprobe.__version__,
            )
        ]
    if reference_capacity is None or soc_anchor is None:
        return [
            _provider_unavailable_record(
                record_id=f"response.pulse_resistance:{cycle_id}:unavailable",
                record_type="response.pulse_resistance",
                cell_id=cell_id,
                cycle_id=cycle_id,
                provider="PyProBE",
                method="pulsing.get_resistances",
                reason=(
                    "no structurally complete cycle capacity coordinate for SOC anchoring"
                    if soc_anchor is None
                    else "no nominal capacity or structurally complete reference charge"
                ),
                version=pyprobe.__version__,
            )
        ]
    records: list[dict[str, Any]] = []
    for pulse_number, (rest, pulse) in enumerate(candidates, start=1):
        start, end = rest["start"], pulse["end"]
        window = selected_py.slice(start, end - start)
        capacity = window["Capacity [Ah]"].cast(pl.Float64).to_numpy()
        soc = np.clip((capacity - soc_anchor) / reference_capacity, 0.0, 1.0)
        pulse_input = window.with_columns(pl.Series("SOC", soc))
        columns = ["Time [s]", "Current [A]", "Voltage [V]", "Event", "SOC", "Capacity [Ah]"]
        try:
            result = pulsing.get_resistances(
                pyprobe_result(pulse_input, columns),
                r_times=list(config.pulse_resistance_times_s),
            ).data
            row = result.row(0, named=True)
            provider_calls.append(
                {
                    "provider": "PyProBE",
                    "method": "pulsing.get_resistances",
                    "status": "ok",
                    "source_interval": {
                        "start_row": rest["source_interval"]["start_row"],
                        "end_row": pulse["source_interval"]["end_row"],
                    },
                }
            )
            metrics = {
                "ocv": metric(json_number(row.get("OCV [V]")), "V"),
                "soc": metric(json_number(row.get("SOC")), "1"),
                "r0": metric(json_number(row.get("R0 [Ohms]")), "ohm"),
                "pulse_current": metric(
                    json_number(
                        np.nanmedian(window["Current [A]"].to_numpy()[-pulse["sample_count"] :])
                    ),
                    "A",
                ),
            }
            for time_s in config.pulse_resistance_times_s:
                value = _resistance_value(row, time_s)
                metrics[f"r_{seconds_label(time_s)}s"] = metric(value, "ohm")
            records.append(
                make_record(
                    record_id=f"response.pulse_resistance:{cycle_id}:{pulse_number}",
                    record_type="response.pulse_resistance",
                    cell_id=cell_id,
                    cycle_scope=cycle_id,
                    source_intervals=[
                        {
                            "start_row": rest["source_interval"]["start_row"],
                            "end_row": pulse["source_interval"]["end_row"],
                            "start_time_s": rest["source_interval"]["start_time_s"],
                            "end_time_s": pulse["source_interval"]["end_time_s"],
                        }
                    ],
                    attributes={
                        "pulse_number": pulse_number,
                        "soc_reference_capacity_ah": reference_capacity,
                        "soc_capacity_coordinate_zero_ah": soc_anchor,
                        "soc_reference_method": "complete_cycle_capacity_coordinate_minimum",
                    },
                    metrics=metrics,
                    provider="PyProBE",
                    method_name="pulsing.get_resistances",
                    provider_version=pyprobe.__version__,
                    parameters={
                        "analysis_policy_version": ANALYSIS_POLICY_VERSION,
                        "r_times_s": list(config.pulse_resistance_times_s),
                        "minimum_preceding_rest_s": config.policy(
                            "current_step_rest_min_s"
                        ),
                        "maximum_pulse_duration_s": config.policy(
                            "current_step_pulse_max_s"
                        ),
                    },
                    references=[PYPROBE_REFERENCE],
                    interpretation_limits=[
                        "Resistance values are method- and pulse-window-specific."
                    ],
                )
            )
        except Exception as exc:  # noqa: BLE001 - provider errors are audit data
            provider_calls.append(
                {
                    "provider": "PyProBE",
                    "method": "pulsing.get_resistances",
                    "status": "error",
                    "error": str(exc),
                    "source_interval": pulse["source_interval"],
                }
            )
            records.append(
                _provider_unavailable_record(
                    record_id=f"response.pulse_resistance:{cycle_id}:{pulse_number}",
                    record_type="response.pulse_resistance",
                    cell_id=cell_id,
                    cycle_id=cycle_id,
                    provider="PyProBE",
                    method="pulsing.get_resistances",
                    reason=f"provider_error: {exc}",
                    version=pyprobe.__version__,
                    source_interval=pulse["source_interval"],
                )
            )
    return records


def analyze_ica_dva(
    frame: pl.DataFrame,
    pyprobe_frame: pl.DataFrame,
    phases: list[dict[str, Any]],
    cycle_summaries: list[dict[str, Any]],
    *,
    config: AnalysisConfig,
    cell_id: str,
    cycle_id: int | None,
    provider_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create complete PyProBE LEAN ICA and DVA curves for eligible phases."""

    if cycle_id is None:
        reason = "no_structurally_complete_representative_cycle"
        return [
            _provider_unavailable_record(
                record_id=f"response.{kind}_curve:None:unavailable",
                record_type=f"response.{kind}_curve",
                cell_id=cell_id,
                cycle_id=None,
                provider="PyProBE",
                method="differentiation.differentiate_lean",
                reason=reason,
                version=pyprobe.__version__,
            )
            for kind in ("ica", "dva")
        ]

    selected = _select_cycle(frame, cycle_id)
    selected_py = _select_py_cycle(pyprobe_frame, cycle_id)
    reference_capacity = config.nominal_capacity_ah or _reference_capacity(cycle_summaries)
    active = [item for item in phases if item["phase"] in {"charge", "discharge"}]
    selected_segments = []
    for phase in ("charge", "discharge"):
        matches = [item for item in active if item["phase"] == phase]
        if matches:
            selected_segments.append(max(matches, key=lambda item: item["duration_s"]))
    records: list[dict[str, Any]] = []
    for segment in selected_segments:
        sub = selected.slice(segment["start"], segment["end"] - segment["start"])
        py_sub = selected_py.slice(segment["start"], segment["end"] - segment["start"])
        reasons = _diagnostic_ineligibility(sub, py_sub, reference_capacity, config=config)
        for kind in ("ica", "dva"):
            record_type = f"response.{kind}_curve"
            record_id = f"{record_type}:{cycle_id}:{segment['phase']}"
            if reasons:
                records.append(
                    _provider_unavailable_record(
                        record_id=record_id,
                        record_type=record_type,
                        cell_id=cell_id,
                        cycle_id=cycle_id,
                        provider="PyProBE",
                        method="differentiation.differentiate_lean",
                        reason="; ".join(reasons),
                        version=pyprobe.__version__,
                        source_interval=segment["source_interval"],
                    )
                )
                continue
            try:
                columns = ["Voltage [V]", "Capacity [Ah]"]
                result_input = pyprobe_result(py_sub, columns)
                if kind == "ica":
                    x_name, y_name = "Voltage [V]", "Capacity [Ah]"
                    derivative = "d(Capacity [Ah])/d(Voltage [V])"
                    x_unit, y_unit = "V", "Ah/V"
                else:
                    x_name, y_name = "Capacity [Ah]", "Voltage [V]"
                    derivative = "d(Voltage [V])/d(Capacity [Ah])"
                    x_unit, y_unit = "Ah", "V/Ah"
                result = differentiation.differentiate_lean(
                    result_input,
                    x_name,
                    y_name,
                    k=1,
                    section="longest",
                ).data
                x = result[x_name].cast(pl.Float64).to_numpy()
                y = result[derivative].cast(pl.Float64).to_numpy()
                finite = np.isfinite(x) & np.isfinite(y)
                x, y = x[finite], y[finite]
                provider_calls.append(
                    {
                        "provider": "PyProBE",
                        "method": "differentiation.differentiate_lean",
                        "status": "ok",
                        "kind": kind,
                        "source_interval": segment["source_interval"],
                    }
                )
                peak_error = None
                try:
                    peaks = _curve_peaks(
                        x,
                        y,
                        prominence_fraction=config.policy(
                            "diagnostic_peak_prominence_fraction"
                        ),
                        minimum_distance_points=int(
                            config.policy("diagnostic_peak_minimum_distance_points")
                        ),
                    )
                    provider_calls.append(
                        {
                            "provider": "SciPy",
                            "method": "signal.find_peaks",
                            "status": "ok",
                            "kind": kind,
                            "source_interval": segment["source_interval"],
                            "peak_count": len(peaks),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - provider audit boundary
                    peaks = []
                    peak_error = str(exc)
                    provider_calls.append(
                        {
                            "provider": "SciPy",
                            "method": "signal.find_peaks",
                            "status": "error",
                            "kind": kind,
                            "error": peak_error,
                            "source_interval": segment["source_interval"],
                        }
                    )
                records.append(
                    make_record(
                        record_id=record_id,
                        record_type=record_type,
                        cell_id=cell_id,
                        cycle_scope=cycle_id,
                        source_intervals=[segment["source_interval"]],
                        attributes={"phase": segment["phase"], "peaks": peaks},
                        metrics={
                            "point_count": metric(len(x), "1"),
                            "peak_count": metric(
                                len(peaks) if peak_error is None else None,
                                "1",
                                status=("ok" if peak_error is None else "not_computable"),
                                reason=(
                                    None if peak_error is None else f"provider_error: {peak_error}"
                                ),
                            ),
                        },
                        series={
                            "x": [float(value) for value in x],
                            "x_name": x_name,
                            "x_unit": x_unit,
                            "y": [float(value) for value in y],
                            "y_name": derivative,
                            "y_unit": y_unit,
                        },
                        provider="PyProBE+SciPy",
                        method_name="differentiation.differentiate_lean",
                        provider_version=(
                            f"PyProBE {pyprobe.__version__}; SciPy {scipy.__version__}"
                        ),
                        parameters={
                            "analysis_policy_version": ANALYSIS_POLICY_VERSION,
                            "k": 1,
                            "section": "longest",
                            "smoothing_filter": [0.0668, 0.2417, 0.383, 0.2417, 0.0668],
                            "peak_prominence_fraction": config.policy(
                                "diagnostic_peak_prominence_fraction"
                            ),
                            "peak_minimum_distance_points": int(
                                config.policy("diagnostic_peak_minimum_distance_points")
                            ),
                        },
                        references=[LEAN_REFERENCE, PYPROBE_REFERENCE],
                        quality_status=("warning" if peak_error is not None else "ok"),
                        quality_flags=(["peak_provider_error"] if peak_error is not None else []),
                        interpretation_limits=[
                            "Curve and peak values are conditional on the eligibility and LEAN parameters recorded here."
                        ],
                    )
                )
            except Exception as exc:  # noqa: BLE001 - provider errors are audit data
                provider_calls.append(
                    {
                        "provider": "PyProBE",
                        "method": "differentiation.differentiate_lean",
                        "status": "error",
                        "kind": kind,
                        "error": str(exc),
                        "source_interval": segment["source_interval"],
                    }
                )
                records.append(
                    _provider_unavailable_record(
                        record_id=record_id,
                        record_type=record_type,
                        cell_id=cell_id,
                        cycle_id=cycle_id,
                        provider="PyProBE",
                        method="differentiation.differentiate_lean",
                        reason=f"provider_error: {exc}",
                        version=pyprobe.__version__,
                        source_interval=segment["source_interval"],
                    )
                )
    if not records:
        for kind in ("ica", "dva"):
            records.append(
                _provider_unavailable_record(
                    record_id=f"response.{kind}_curve:{cycle_id}:unavailable",
                    record_type=f"response.{kind}_curve",
                    cell_id=cell_id,
                    cycle_id=cycle_id,
                    provider="PyProBE",
                    method="differentiation.differentiate_lean",
                    reason="no_charge_or_discharge_phase_in_representative_scope",
                    version=pyprobe.__version__,
                )
            )
    return records


def _cycle_record(
    frame: pl.DataFrame,
    py_frame: pl.DataFrame,
    *,
    cycle_id: int,
    cycle_id_source: str,
    adapter_details: dict[str, Any],
    touches_dataset_start: bool,
    touches_dataset_end: bool,
    config: AnalysisConfig,
    cell_id: str,
    provider_calls: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    time = frame["test_time_s"].cast(pl.Float64).to_numpy()
    current = frame["current_a"].cast(pl.Float64).to_numpy()
    voltage = frame["voltage_v"].cast(pl.Float64).to_numpy()
    interval = previous_intervals(
        time,
        current,
        sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
    )
    coverage = finite_coverage(time, current, voltage)
    minimum_coverage = config.policy("cycle_integration_coverage_min")
    efficiency_warning_max = config.policy("cycle_efficiency_warning_max")
    has_charge = bool(np.any(current > config.rest_current_threshold_a))
    has_discharge = bool(np.any(current < -config.rest_current_threshold_a))
    structural_flags: list[str] = []
    warning_flags: list[str] = []
    if cycle_id_source not in {"source", "joined"}:
        structural_flags.append(f"cycle_id_{cycle_id_source}")
    if not has_charge:
        structural_flags.append("charge_phase_missing")
    if not has_discharge:
        structural_flags.append("discharge_phase_missing")
    if coverage < minimum_coverage:
        structural_flags.append("integration_coverage_below_policy_minimum")
    if interval.non_positive_interval_count:
        structural_flags.append("non_positive_time_interval")
    if interval.sampling_interval_outlier_count:
        warning_flags.append("sampling_interval_outlier")
    if touches_dataset_start and len(current) and abs(current[0]) > config.rest_current_threshold_a:
        structural_flags.append("possible_truncation_at_dataset_start")
    if touches_dataset_end and len(current) and abs(current[-1]) > config.rest_current_threshold_a:
        structural_flags.append("possible_truncation_at_dataset_end")

    capacity_input_method = str(adapter_details["capacity_method"])
    capacity_flags = list(adapter_details["capacity_flags"])
    reported_capacity_finite_coverage_min = float(
        adapter_details["reported_capacity_finite_coverage_min"]
    )
    charge_capacity = discharge_capacity = throughput = None
    provider_error: str | None = None
    try:
        result = cycling.summary(pyprobe_experiment(py_frame)).data
        row = result.row(0, named=True)
        charge_capacity = json_number(row.get("Charge Capacity [Ah]"))
        discharge_capacity = json_number(row.get("Discharge Capacity [Ah]"))
        throughput = json_number(row.get("Capacity Throughput [Ah]"))
        provider_calls.append(
            {
                "provider": "PyProBE",
                "method": "cycling.summary",
                "status": "ok",
                "cycle": cycle_id,
                "capacity_input_method": capacity_input_method,
                "source_interval": _frame_source_interval(frame),
            }
        )
    except Exception as exc:  # noqa: BLE001 - provider errors are audit data
        provider_error = str(exc)
        structural_flags.append("provider_error")
        provider_calls.append(
            {
                "provider": "PyProBE",
                "method": "cycling.summary",
                "status": "error",
                "cycle": cycle_id,
                "error": provider_error,
                "source_interval": _frame_source_interval(frame),
            }
        )

    charge_energy, charge_energy_method, charge_energy_flags = _phase_energy(
        frame,
        positive=True,
        config=config,
    )
    discharge_energy, discharge_energy_method, discharge_energy_flags = _phase_energy(
        frame,
        positive=False,
        config=config,
    )
    warning_flags.extend(capacity_flags + charge_energy_flags + discharge_energy_flags)
    structurally_complete = (
        not structural_flags and charge_capacity is not None and discharge_capacity is not None
    )
    coulombic = (
        discharge_capacity / charge_capacity if structurally_complete and charge_capacity else None
    )
    energy_efficiency = (
        discharge_energy / charge_energy if structurally_complete and charge_energy else None
    )
    if coulombic is not None and not 0 <= coulombic <= efficiency_warning_max:
        warning_flags.append("coulombic_efficiency_outside_policy_range")
    if energy_efficiency is not None and not 0 <= energy_efficiency <= efficiency_warning_max:
        warning_flags.append("energy_efficiency_outside_policy_range")
    flags = sorted(set(structural_flags + warning_flags))

    rows = frame["_source_row"].to_numpy()
    records = frame["record_index"].to_numpy() if "record_index" in frame.columns else rows
    source_interval = {
        "start_row": int(rows[0]),
        "end_row": int(rows[-1]),
        "start_record": int(records[0]),
        "end_record": int(records[-1]),
        "start_time_s": float(time[0]),
        "end_time_s": float(time[-1]),
    }
    metrics = {
        "charge_capacity": metric(
            charge_capacity,
            "Ah",
            status="not_computable" if charge_capacity is None else "ok",
            reason=provider_error,
        ),
        "discharge_capacity": metric(
            discharge_capacity,
            "Ah",
            status="not_computable" if discharge_capacity is None else "ok",
            reason=provider_error,
        ),
        "capacity_throughput": metric(throughput, "Ah"),
        "charge_energy": metric(charge_energy, "Wh"),
        "discharge_energy": metric(discharge_energy, "Wh"),
        "coulombic_efficiency": metric(
            coulombic,
            "1",
            status="ok" if coulombic is not None else "not_computable",
            reason=None if coulombic is not None else "cycle is not structurally complete",
        ),
        "energy_efficiency": metric(
            energy_efficiency,
            "1",
            status="ok" if energy_efficiency is not None else "not_computable",
            reason=None if energy_efficiency is not None else "cycle is not structurally complete",
        ),
        "integration_coverage": metric(coverage, "1"),
        "duration": metric(float(time[-1] - time[0]) if len(time) else None, "s"),
    }
    record = make_record(
        record_id=f"response.cycle_summary:{cycle_id}",
        record_type="response.cycle_summary",
        cell_id=cell_id,
        cycle_scope=cycle_id,
        source_intervals=[source_interval],
        attributes={
            "cycle_id_source": cycle_id_source,
            "structurally_complete": structurally_complete,
            "capacity_input_method": capacity_input_method,
            "charge_energy_method": charge_energy_method,
            "discharge_energy_method": discharge_energy_method,
        },
        metrics=metrics,
        provider="PyProBE+BFL",
        method_name="cycling.summary_with_structural_completeness_v1",
        provider_version=f"PyProBE {pyprobe.__version__}; BFL 0.4.0",
        parameters={
            "analysis_policy_version": ANALYSIS_POLICY_VERSION,
            "minimum_integration_coverage": minimum_coverage,
            "efficiency_warning_maximum": efficiency_warning_max,
            "reported_capacity_finite_coverage_minimum": (
                reported_capacity_finite_coverage_min
            ),
            "sampling_interval_outlier_factor": (config.sampling_interval_outlier_factor),
        },
        references=[PYPROBE_REFERENCE],
        applicability_status="applicable" if structurally_complete else "partial",
        applicability_reasons=[] if structurally_complete else structural_flags,
        quality_status="warning" if flags else "ok",
        quality_flags=sorted(set(flags)),
        interpretation_limits=[
            "Completeness is structural and does not assert conformance to a declared protocol."
        ],
    )
    summary = {
        "cycle_id": cycle_id,
        "cycle_id_source": cycle_id_source,
        "complete": structurally_complete,
        "charge_capacity_ah": charge_capacity,
        "discharge_capacity_ah": discharge_capacity,
        "charge_energy_wh": charge_energy,
        "discharge_energy_wh": discharge_energy,
        "operation_signature": _operation_signature(current, config.rest_current_threshold_a),
        "capacity_coordinate_min_ah": (
            float(np.nanmin(py_frame["Capacity [Ah]"].to_numpy())) if py_frame.height else None
        ),
        "capacity_coordinate_max_ah": (
            float(np.nanmax(py_frame["Capacity [Ah]"].to_numpy())) if py_frame.height else None
        ),
        "source_interval": source_interval,
    }
    return record, summary


def _phase_energy(
    frame: pl.DataFrame,
    *,
    positive: bool,
    config: AnalysisConfig,
) -> tuple[float | None, str, list[str]]:
    name = "charge_energy_wh" if positive else "discharge_energy_wh"
    current = frame["current_a"].cast(pl.Float64).to_numpy()
    mask = (
        current > config.rest_current_threshold_a
        if positive
        else current < -config.rest_current_threshold_a
    )
    subset = frame.filter(pl.Series(mask))
    if name in subset.columns:
        delta, flags = reported_delta(subset[name].cast(pl.Float64, strict=False).to_numpy())
        if delta is not None:
            return delta, "reported_column_delta", []
    else:
        flags = ["reported_energy_column_missing"]
    if not np.any(mask):
        return None, "zoh_previous_v1", flags + ["phase_missing"]
    time = frame["test_time_s"].cast(pl.Float64).to_numpy()
    voltage = frame["voltage_v"].cast(pl.Float64).to_numpy()
    power = np.where(mask, np.abs(voltage * current), np.nan)
    value, interval = integrate_previous(
        time,
        power,
        scale=1.0 / 3600.0,
        sampling_interval_outlier_factor=(config.sampling_interval_outlier_factor),
    )
    if interval.sampling_interval_outlier_count:
        flags.append("zoh_energy_includes_sampling_interval_outlier")
    return value, "zoh_previous_v1", flags


def _operation_signature(current: np.ndarray, threshold: float) -> str:
    phase = np.where(np.abs(current) <= threshold, "R", np.where(current > 0, "C", "D"))
    compressed = [str(phase[0])] if len(phase) else []
    for value in phase[1:]:
        if value != compressed[-1]:
            compressed.append(str(value))
    return "-".join(compressed)


def _reference_capacity(summaries: list[dict[str, Any]]) -> float | None:
    values = [
        item["charge_capacity_ah"]
        for item in summaries
        if item.get("complete") and item.get("charge_capacity_ah")
    ]
    return float(np.median(values[:4])) if values else None


def _soc_anchor(summaries: list[dict[str, Any]], cycle_id: int | None) -> float | None:
    """Return a capacity-coordinate zero only from the selected complete cycle."""

    for item in summaries:
        if (
            item.get("complete")
            and item.get("cycle_id") == cycle_id
            and item.get("capacity_coordinate_min_ah") is not None
        ):
            return float(item["capacity_coordinate_min_ah"])
    return None


def _pulse_candidates(
    phases: list[dict[str, Any]],
    required_time: float,
    *,
    rest_min_s: float,
    pulse_max_s: float,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    output = []
    for index, segment in enumerate(phases):
        if index == 0 or segment["phase"] == "rest":
            continue
        rest = phases[index - 1]
        if (
            rest["phase"] == "rest"
            and rest["duration_s"] >= rest_min_s
            and segment["duration_s"] <= pulse_max_s
            and segment["duration_s"] >= required_time
            and rest["duration_s"] >= segment["duration_s"]
        ):
            output.append((rest, segment))
    return output


def _diagnostic_ineligibility(
    frame: pl.DataFrame,
    py_frame: pl.DataFrame,
    reference_capacity: float | None,
    *,
    config: AnalysisConfig,
) -> list[str]:
    reasons: list[str] = []
    minimum_samples = int(config.policy("diagnostic_min_samples"))
    minimum_coverage = config.policy("diagnostic_finite_coverage_min")
    monotone_fraction = config.policy("diagnostic_monotone_fraction_min")
    minimum_voltage_span = config.policy("diagnostic_min_voltage_span_v")
    maximum_c_rate = config.policy("diagnostic_max_c_rate")
    if frame.height < minimum_samples:
        reasons.append("fewer_samples_than_diagnostic_policy_minimum")
        return reasons
    time = frame["test_time_s"].cast(pl.Float64).to_numpy()
    current = frame["current_a"].cast(pl.Float64).to_numpy()
    voltage = frame["voltage_v"].cast(pl.Float64).to_numpy()
    capacity = py_frame["Capacity [Ah]"].cast(pl.Float64).to_numpy()
    if finite_coverage(time, current, voltage) < minimum_coverage:
        reasons.append("valid_coverage_below_diagnostic_policy_minimum")
    if float(np.nanmax(voltage) - np.nanmin(voltage)) < minimum_voltage_span:
        reasons.append("voltage_span_below_diagnostic_policy_minimum")
    direction_increasing = bool(voltage[-1] >= voltage[0])
    if not mostly_monotone(
        voltage,
        increasing=direction_increasing,
        fraction=monotone_fraction,
    ):
        reasons.append("voltage_monotonicity_below_diagnostic_policy_minimum")
    capacity_increasing = bool(capacity[-1] >= capacity[0])
    if not mostly_monotone(
        capacity,
        increasing=capacity_increasing,
        fraction=monotone_fraction,
    ):
        reasons.append("capacity_monotonicity_below_diagnostic_policy_minimum")
    if reference_capacity is None or reference_capacity <= 0:
        reasons.append("reference_capacity_unavailable")
    elif float(np.nanmedian(np.abs(current)) / reference_capacity) > maximum_c_rate:
        reasons.append("equivalent_c_rate_above_diagnostic_policy_maximum")
    return reasons


def _resistance_value(row: dict[str, Any], time_s: float) -> float | int | None:
    candidates = (
        f"R_{seconds_label(time_s)}s [Ohms]",
        f"R_{time_s}s [Ohms]",
        f"R_{time_s:g}s [Ohms]",
        f"R_{float(time_s):.1f}s [Ohms]",
    )
    for name in candidates:
        if name in row:
            return json_number(row[name])
    return None


def _curve_peaks(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prominence_fraction: float,
    minimum_distance_points: int,
) -> list[dict[str, float]]:
    if len(y) < 3:
        return []
    magnitude = np.abs(y)
    q05, q95 = np.quantile(magnitude[np.isfinite(magnitude)], [0.05, 0.95])
    prominence = max(float((q95 - q05) * prominence_fraction), np.finfo(float).eps)
    indexes, _ = find_peaks(
        magnitude,
        prominence=prominence,
        distance=minimum_distance_points,
    )
    prominences = peak_prominences(magnitude, indexes)[0] if len(indexes) else []
    return [
        {"x": float(x[index]), "value": float(y[index]), "prominence": float(prominences[pos])}
        for pos, index in enumerate(indexes)
    ]


def _interpolate_checkpoint(
    relative_time: np.ndarray,
    values: np.ndarray,
    checkpoint_s: float,
    *,
    sampling_interval_outlier_factor: float,
) -> tuple[float | None, float | None, float | None, str | None]:
    """Interpolate only within finite, increasing, locally supported samples."""

    if len(relative_time) < 2 or len(values) < 2:
        return None, None, None, "fewer_than_two_finite_rest_voltage_samples"
    deltas = np.diff(relative_time)
    if np.any(~np.isfinite(deltas)) or np.any(deltas <= 0):
        return None, None, None, "non_positive_rest_time_interval"
    if checkpoint_s < relative_time[0] or checkpoint_s > relative_time[-1]:
        return None, None, None, "rest_interval_does_not_reach_checkpoint"

    insertion = int(np.searchsorted(relative_time, checkpoint_s, side="left"))
    if insertion < len(relative_time) and np.isclose(relative_time[insertion], checkpoint_s):
        return (
            float(values[insertion]),
            0.0,
            abs(float(relative_time[insertion] - checkpoint_s)),
            None,
        )
    left = insertion - 1
    right = insertion
    bracket_width = float(relative_time[right] - relative_time[left])
    adjacent = []
    if left - 1 >= 0:
        adjacent.append(float(deltas[left - 1]))
    if left + 1 < len(deltas):
        adjacent.append(float(deltas[left + 1]))
    local_reference = max(adjacent) if adjacent else float(np.median(deltas))
    if bracket_width > sampling_interval_outlier_factor * local_reference:
        return None, None, None, "checkpoint_sampling_interval_too_sparse"
    nearest_offset = float(
        min(
            abs(relative_time[left] - checkpoint_s),
            abs(relative_time[right] - checkpoint_s),
        )
    )
    return (
        float(np.interp(checkpoint_s, relative_time, values)),
        bracket_width,
        nearest_offset,
        None,
    )


def _select_cycle(frame: pl.DataFrame, cycle_id: int | None) -> pl.DataFrame:
    if cycle_id is None or "cycle_index" not in frame.columns:
        return frame.sort("test_time_s")
    return frame.filter(pl.col("cycle_index") == cycle_id).sort("test_time_s")


def _frame_source_interval(frame: pl.DataFrame) -> dict[str, Any]:
    """Return the exact canonical scope passed to a third-party provider."""

    rows = frame["_source_row"].to_numpy()
    records = frame["record_index"].to_numpy() if "record_index" in frame.columns else rows
    time = frame["test_time_s"].cast(pl.Float64).to_numpy()
    return {
        "start_row": int(rows[0]),
        "end_row": int(rows[-1]),
        "start_record": int(records[0]),
        "end_record": int(records[-1]),
        "start_time_s": float(time[0]),
        "end_time_s": float(time[-1]),
    }


def _select_py_cycle(frame: pl.DataFrame, cycle_id: int | None) -> pl.DataFrame:
    if cycle_id is None:
        return frame
    return frame.filter(pl.col("Cycle") == cycle_id).sort("Time [s]")


def _cycle_unavailable_record(cell_id: str) -> dict[str, Any]:
    return _provider_unavailable_record(
        record_id="response.cycle_summary:unavailable",
        record_type="response.cycle_summary",
        cell_id=cell_id,
        cycle_id=None,
        provider="BDS",
        method="cycle_index",
        reason="BDS output has no cycle_index",
        version=getattr(bds, "__version__", "unknown"),
    )


def _provider_unavailable_record(
    *,
    record_id: str,
    record_type: str,
    cell_id: str,
    cycle_id: int | None,
    provider: str,
    method: str,
    reason: str,
    version: str,
    source_interval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return make_record(
        record_id=record_id,
        record_type=record_type,
        cell_id=cell_id,
        cycle_scope=cycle_id,
        source_intervals=[source_interval] if source_interval else [],
        provider=provider,
        method_name=method,
        provider_version=version,
        applicability_status="not_computable",
        applicability_reasons=[reason],
        quality_status="warning",
        quality_flags=["provider_error"]
        if reason.startswith("provider_error")
        else ["input_not_eligible"],
        interpretation_limits=["No same-name local fallback was used."],
    )


def record_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count records by stable record type."""

    return dict(sorted(Counter(record["record_type"] for record in records).items()))
