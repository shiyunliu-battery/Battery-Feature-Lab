"""Conservative cycle-level capacity comparison and evolution summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import scipy
from scipy.stats import kendalltau, theilslopes

from battery_feature_lab.analysis.schema import AnalysisConfig, make_record, metric

THEIL_REFERENCE = (
    "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html"
)
KENDALL_REFERENCE = (
    "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html"
)

MINIMUM_REPEAT_OBSERVATIONS = 2
MINIMUM_REFERENCE_CYCLES = 3
MINIMUM_TREND_CYCLES = 8


def _finite_number(value: Any) -> bool:
    """Return True only for finite numeric values."""
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _cycle_sort_key(item: dict[str, Any]) -> tuple[int, float | str]:
    """Sort finite numeric cycle IDs first without assuming all IDs are valid."""
    value = item.get("cycle_id")
    if _finite_number(value):
        return (0, float(value))
    return (1, str(value))


def _dominant_comparable_group(
    items: list[dict[str, Any]],
) -> tuple[Any | None, list[dict[str, Any]], str | None]:
    """Return one unambiguous operation-signature group."""
    signed = [item for item in items if item.get("operation_signature") is not None]
    if not signed:
        return None, [], "missing_operation_signature"

    counts = Counter(item["operation_signature"] for item in signed)
    largest_group = max(counts.values())
    dominant_signatures = [
        signature for signature, count in counts.items() if count == largest_group
    ]

    if len(dominant_signatures) != 1:
        return None, [], "ambiguous_dominant_operation_signature"

    signature = dominant_signatures[0]
    comparable = [item for item in signed if item["operation_signature"] == signature]
    comparable.sort(key=_cycle_sort_key)
    return signature, comparable, None


def analyze_capacity_evolution(
    summaries: list[dict[str, Any]],
    *,
    config: AnalysisConfig,
    cell_id: str,
    provider_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize comparable cycle-level capacity observations conservatively.

    This function operates only on cycle summaries supplied by the upstream
    cycle analysis. It does not search the raw time series for complete charge
    or discharge phases. Absence of usable cycle summaries therefore does not
    mean that phase-level capacity observations are absent from the source data.
    """

    ordered = sorted(summaries, key=_cycle_sort_key)

    # ------------------------------------------------------------------
    # 1. Find complete cycle-level discharge-capacity observations.
    #
    # A source/joined cycle ID is required here because this function is a
    # cycle-level analysis. Phase-level capacity observations must be supplied
    # by a separate upstream route rather than inferred here.
    # ------------------------------------------------------------------
    source_ordered_observations = [
        item
        for item in ordered
        if _finite_number(item.get("cycle_id"))
        and item.get("cycle_id_source") in {"source", "joined"}
    ]

    # Exclusion is defined on the source ordering, before completeness gates.
    # An incomplete first source cycle must not cause the next complete cycle
    # to be excluded as well.
    exclude_count = max(0, int(config.formation_cycles_to_exclude))
    excluded_items = source_ordered_observations[:exclude_count]
    excluded_ids = {item["cycle_id"] for item in excluded_items}

    complete_capacity_observations = [
        item
        for item in ordered
        if item.get("complete") is True
        and _finite_number(item.get("cycle_id"))
        and _finite_number(item.get("discharge_capacity_ah"))
    ]

    reliable_cycle_observations = [
        item
        for item in complete_capacity_observations
        if item.get("cycle_id_source") in {"source", "joined"}
    ]

    after_exclusion = [
        item for item in reliable_cycle_observations if item["cycle_id"] not in excluded_ids
    ]

    # ------------------------------------------------------------------
    # 3. Form one conservative comparable group.
    #
    # Missing operation signatures are not treated as evidence that cycles are
    # comparable. A tie between dominant signatures is not resolved silently.
    # ------------------------------------------------------------------
    signature = None
    comparable: list[dict[str, Any]] = []
    comparability_reason = None

    if after_exclusion:
        signature, comparable, comparability_reason = _dominant_comparable_group(after_exclusion)

    # Duplicate cycle IDs make ordering and independent repeat counts
    # ambiguous, so do not use them for comparative metrics.
    cycle_ids = [item["cycle_id"] for item in comparable]
    duplicate_cycle_ids = sorted(
        cycle_id for cycle_id, count in Counter(cycle_ids).items() if count > 1
    )
    cycle_id_reason = "duplicate_cycle_ids_in_comparable_group" if duplicate_cycle_ids else None

    analyzed = comparable if cycle_id_reason is None else []

    # ------------------------------------------------------------------
    # 4. Record-level applicability.
    #
    # These reasons describe this cycle-level route only. They deliberately do
    # not claim that complete charge/discharge phases are absent from raw data.
    # ------------------------------------------------------------------
    applicability_reasons: list[str] = []

    if not summaries:
        applicability_reasons.append("no_cycle_summaries")
    elif not complete_capacity_observations:
        applicability_reasons.append("no_complete_cycle_level_discharge_capacity_observations")
    elif not reliable_cycle_observations:
        applicability_reasons.append("no_source_or_joined_cycle_level_capacity_observations")
    elif not after_exclusion:
        applicability_reasons.append(
            "all_cycle_level_capacity_observations_excluded_by_configuration"
        )
    elif comparability_reason is not None:
        applicability_reasons.append(comparability_reason)
    elif cycle_id_reason is not None:
        applicability_reasons.append(cycle_id_reason)
    elif len(analyzed) < MINIMUM_REPEAT_OBSERVATIONS:
        applicability_reasons.append("fewer_than_two_comparable_cycle_level_capacity_observations")

    # ------------------------------------------------------------------
    # 5. Descriptive repeat-measurement metrics.
    #
    # These describe the spread of comparable cycle-level capacity values.
    # They do not imply ageing, fade, or state of health.
    # ------------------------------------------------------------------
    comparable_capacities = np.asarray(
        [float(item["discharge_capacity_ah"]) for item in analyzed],
        dtype=float,
    )

    comparable_median = None
    comparable_range = None
    comparable_relative_range = None
    repeatability_reason = None

    if len(comparable_capacities) >= MINIMUM_REPEAT_OBSERVATIONS:
        comparable_median = float(np.median(comparable_capacities))
        comparable_range = float(np.max(comparable_capacities) - np.min(comparable_capacities))
        if comparable_median > 0:
            comparable_relative_range = comparable_range / comparable_median
        else:
            repeatability_reason = "non_positive_comparable_capacity_median"
    else:
        repeatability_reason = "fewer_than_two_comparable_cycle_level_capacity_observations"

    # ------------------------------------------------------------------
    # 6. Last comparable discharge capacity.
    # ------------------------------------------------------------------
    last = None
    last_reason = None

    if analyzed:
        last = float(analyzed[-1]["discharge_capacity_ah"])
    else:
        last_reason = (
            applicability_reasons[0]
            if applicability_reasons
            else "no_comparable_cycle_level_capacity_observations"
        )

    # ------------------------------------------------------------------
    # 7. Reference capacity and relative retention.
    #
    # Retention requires a complete reference window AND at least one later
    # comparable observation. This avoids comparing a reference window with a
    # "last" value that is itself part of that same reference window.
    # ------------------------------------------------------------------
    reference_window_size = max(1, int(config.reference_window_size))
    reference_items = analyzed[:reference_window_size]

    reference = None
    reference_reason = None
    retention = None
    retention_reason = None

    if len(reference_items) < MINIMUM_REFERENCE_CYCLES:
        reference_reason = "fewer_than_three_reference_cycles"
        retention_reason = reference_reason
    else:
        reference = float(
            np.median([float(item["discharge_capacity_ah"]) for item in reference_items])
        )

        if not np.isfinite(reference):
            reference = None
            reference_reason = "non_finite_reference_capacity"
            retention_reason = reference_reason
        elif reference <= 0:
            reference = None
            reference_reason = "non_positive_reference_capacity"
            retention_reason = reference_reason
        elif len(analyzed) <= reference_window_size:
            retention_reason = "no_comparable_cycle_after_reference_window"
        elif last is None:
            retention_reason = last_reason or "last_capacity_not_available"
        else:
            retention = last / reference
            if not np.isfinite(retention):
                retention = None
                retention_reason = "non_finite_capacity_retention"

    # ------------------------------------------------------------------
    # 8. Descriptive trend over source/joined cycle IDs.
    #
    # Theil-Sen reports a robust slope and Kendall tau-b reports monotonic
    # ordering. Neither statistic is interpreted here as evidence of ageing or
    # statistical significance.
    # ------------------------------------------------------------------
    slope = None
    tau = None
    slope_reason = None
    tau_reason = None
    analyzed_source_intervals = [
        item["source_interval"] for item in analyzed if item.get("source_interval") is not None
    ]

    if len(analyzed) < MINIMUM_TREND_CYCLES:
        slope_reason = "fewer_than_eight_comparable_cycles"
        tau_reason = "fewer_than_eight_comparable_cycles"
    else:
        cycles = np.asarray([item["cycle_id"] for item in analyzed], dtype=float)
        capacities = np.asarray([item["discharge_capacity_ah"] for item in analyzed], dtype=float)

        if not np.all(np.isfinite(cycles)):
            slope_reason = "cycle_ids_are_not_finite"
            tau_reason = slope_reason
        elif len(np.unique(cycles)) != len(cycles):
            slope_reason = "cycle_ids_are_not_unique"
            tau_reason = slope_reason
        elif np.any(np.diff(cycles) <= 0):
            slope_reason = "cycle_ids_do_not_define_a_strictly_increasing_axis"
            tau_reason = slope_reason
        else:
            try:
                result = theilslopes(capacities, cycles)
                candidate_slope = float(result.slope * 100.0)

                if np.isfinite(candidate_slope):
                    slope = candidate_slope
                    provider_status = "ok"
                    provider_error = None
                else:
                    slope_reason = "non_finite_theil_sen_result"
                    provider_status = "not_computable"
                    provider_error = slope_reason

                call = {
                    "provider": "SciPy",
                    "method": "stats.theilslopes",
                    "status": provider_status,
                    "cycle_ids": cycles.tolist(),
                    "source_intervals": analyzed_source_intervals,
                }
                if provider_error is not None:
                    call["error"] = provider_error
                provider_calls.append(call)

            except Exception as exc:  # noqa: BLE001 - provider errors are audit data
                slope_reason = f"provider_error: {exc}"
                provider_calls.append(
                    {
                        "provider": "SciPy",
                        "method": "stats.theilslopes",
                        "status": "error",
                        "error": str(exc),
                        "cycle_ids": cycles.tolist(),
                        "source_intervals": analyzed_source_intervals,
                    }
                )

            try:
                result = kendalltau(cycles, capacities, variant="b")
                candidate_tau = float(result.statistic)

                if np.isfinite(candidate_tau):
                    tau = candidate_tau
                    provider_status = "ok"
                    provider_error = None
                else:
                    tau_reason = "non_finite_kendall_tau_result"
                    provider_status = "not_computable"
                    provider_error = tau_reason

                call = {
                    "provider": "SciPy",
                    "method": "stats.kendalltau",
                    "status": provider_status,
                    "variant": "b",
                    "cycle_ids": cycles.tolist(),
                    "source_intervals": analyzed_source_intervals,
                }
                if provider_error is not None:
                    call["error"] = provider_error
                provider_calls.append(call)

            except Exception as exc:  # noqa: BLE001 - provider errors are audit data
                tau_reason = f"provider_error: {exc}"
                provider_calls.append(
                    {
                        "provider": "SciPy",
                        "method": "stats.kendalltau",
                        "status": "error",
                        "error": str(exc),
                        "variant": "b",
                        "cycle_ids": cycles.tolist(),
                        "source_intervals": analyzed_source_intervals,
                    }
                )

    # ------------------------------------------------------------------
    # 9. Record construction.
    # ------------------------------------------------------------------
    source_intervals = analyzed_source_intervals

    metric_reasons = [
        repeatability_reason,
        reference_reason,
        retention_reason,
        slope_reason,
        tau_reason,
    ]

    # Provider failures are quality problems. Expected insufficiency of cycle
    # evidence is an applicability limitation, not automatically a quality
    # defect.
    provider_quality_flags = sorted(
        {
            reason
            for reason in metric_reasons
            if reason is not None and reason.startswith("provider_error:")
        }
    )

    record_applicable = len(analyzed) >= MINIMUM_REPEAT_OBSERVATIONS

    if record_applicable:
        analysis_context = (
            "cycle_level_reference_comparison"
            if retention is not None
            else "cycle_level_repeat_measurement"
        )
    else:
        analysis_context = "insufficient_cycle_level_evidence"

    return make_record(
        record_id="evolution.capacity",
        record_type="evolution.capacity",
        cell_id=cell_id,
        cycle_scope=[item["cycle_id"] for item in analyzed],
        source_intervals=source_intervals,
        attributes={
            "analysis_scope": "cycle_summaries_only",
            "analysis_context": analysis_context,
            "phase_level_capacity_search_performed": False,
            "operation_signature": signature,
            "configured_early_cycles_to_exclude": exclude_count,
            "excluded_cycle_ids": sorted(excluded_ids),
            "duplicate_cycle_ids": duplicate_cycle_ids,
            "reference_cycle_ids": [item["cycle_id"] for item in reference_items],
            "comparable_cycle_ids": [item["cycle_id"] for item in analyzed],
            "cycle_id_source_requirement": ["source", "joined"],
            "retention_definition": (
                "last comparable discharge capacity divided by the median "
                "discharge capacity of the configured reference window"
            ),
        },
        metrics={
            "comparable_capacity_median": metric(
                comparable_median,
                "Ah",
                status=("ok" if comparable_median is not None else "not_computable"),
                reason=repeatability_reason,
            ),
            "comparable_capacity_range": metric(
                comparable_range,
                "Ah",
                status=("ok" if comparable_range is not None else "not_computable"),
                reason=repeatability_reason,
            ),
            "comparable_capacity_relative_range": metric(
                comparable_relative_range,
                "1",
                status=("ok" if comparable_relative_range is not None else "not_computable"),
                reason=repeatability_reason,
            ),
            "reference_discharge_capacity": metric(
                reference,
                "Ah",
                status="ok" if reference is not None else "not_computable",
                reason=reference_reason,
            ),
            "last_discharge_capacity": metric(
                last,
                "Ah",
                status="ok" if last is not None else "not_computable",
                reason=last_reason,
            ),
            "capacity_retention": metric(
                retention,
                "1",
                status="ok" if retention is not None else "not_computable",
                reason=retention_reason,
            ),
            "theil_sen_slope_per_100_cycles": metric(
                slope,
                "Ah/100 cycles",
                status="ok" if slope is not None else "not_computable",
                reason=slope_reason,
            ),
            "kendall_tau_b": metric(
                tau,
                "1",
                status="ok" if tau is not None else "not_computable",
                reason=tau_reason,
            ),
            "complete_cycle_capacity_observation_count": metric(
                len(complete_capacity_observations),
                "1",
            ),
            "reliable_cycle_capacity_observation_count": metric(
                len(reliable_cycle_observations),
                "1",
            ),
            "comparable_cycle_count": metric(
                len(analyzed),
                "1",
            ),
        },
        series={},
        provider="SciPy+BFL",
        method_name="conservative_cycle_level_capacity_evolution_v2",
        provider_version=f"SciPy {scipy.__version__}; BFL 0.4.0",
        parameters={
            "reference_window_size": reference_window_size,
            "minimum_repeat_observations": MINIMUM_REPEAT_OBSERVATIONS,
            "minimum_reference_cycles": MINIMUM_REFERENCE_CYCLES,
            "minimum_trend_cycles": MINIMUM_TREND_CYCLES,
            "kendall_variant": "b",
            "trend_axis": "source_or_joined_cycle_id",
        },
        references=[THEIL_REFERENCE, KENDALL_REFERENCE],
        applicability_status=("applicable" if record_applicable else "not_computable"),
        applicability_reasons=([] if record_applicable else applicability_reasons),
        quality_status="warning" if provider_quality_flags else "ok",
        quality_flags=provider_quality_flags,
        interpretation_limits=[
            "This record uses cycle summaries only and does not search the raw time series for complete charge/discharge phases.",
            "Comparable-capacity spread describes repeat measurement at the cycle-summary level and is not an ageing, capacity-fade, or state-of-health claim.",
            "Capacity retention is a relative reference-window comparison and is not, by itself, evidence of ageing or state of health.",
            "Configured early-cycle exclusion is an analysis setting and does not establish that an excluded cycle is a formation cycle.",
            "Operation-signature matching is the comparability rule used by this method; it does not prove that all unrepresented operating conditions are identical.",
            "The Theil-Sen and Kendall statistics are descriptive. No statistical-significance or causal claim is made.",
            "Trend calculations use source or joined cycle_id values as the ordering axis; this function does not independently verify the physical meaning of that source identifier.",
        ],
    )
