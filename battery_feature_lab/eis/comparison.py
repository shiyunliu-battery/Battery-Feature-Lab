"""Question-dependent comparison gates and observed spectral differences."""

from __future__ import annotations

import numpy as np

from battery_feature_lab.analysis.schema import make_record, metric
from battery_feature_lab.eis.analysis import interpolate_impedance
from battery_feature_lab.eis.schema import EISConfig


def compare_spectra(spectra, records, refs, config: EISConfig):
    result = []
    for left_name, right_name in config.comparison_pairs:
        pair = []
        for name in (left_name, right_name):
            matches = [s for s in spectra if name in {s["spectrum_id"], s["spectrum_uid"]}]
            if len(matches) != 1:
                raise ValueError(f"Comparison spectrum ID is missing or ambiguous: {name}")
            pair.append(matches[0])
        left, right = pair
        mismatches, missing = [], []
        fields = [
            "cell_id",
            "temperature_deg_c",
            "soc_fraction",
            "rest_time_s",
            "instrument",
            "protocol_id",
        ]
        amplitude = (
            "ac_amplitude_v"
            if all(s["conditions"]["ac_amplitude_v"]["value"] is not None for s in pair)
            else "ac_amplitude_a"
        )
        fields.append(amplitude)
        axis = {"soc": "soc_fraction", "temperature": "temperature_deg_c"}.get(
            config.comparison_kind
        )
        if axis:
            fields.remove(axis)
        for field in fields:
            values = [s["conditions"][field]["value"] for s in pair]
            if None in values:
                missing.append(field)
            elif isinstance(values[0], (int, float)) and isinstance(values[1], (int, float)):
                if abs(values[0] - values[1]) > config.comparison_tolerances.get(field, 0.0):
                    mismatches.append(field)
            elif values[0] != values[1]:
                mismatches.append(field)
        if axis and any(s["conditions"][axis]["value"] is None for s in pair):
            missing.append(axis)
        elif axis and left["conditions"][axis]["value"] == right["conditions"][axis]["value"]:
            mismatches.append("comparison_axis_unchanged")
        if config.comparison_kind == "ageing":
            order_field = next(
                (
                    f
                    for f in ("cycle_index", "elapsed_time_s", "sequence_index")
                    if all(s["conditions"][f]["value"] is not None for s in pair)
                ),
                None,
            )
            if order_field is None:
                missing.append("recorded_ageing_order")
            elif (
                left["conditions"][order_field]["value"]
                >= right["conditions"][order_field]["value"]
            ):
                mismatches.append("recorded_ageing_order")
        arrays = []
        for spectrum in pair:
            frame = spectrum["frame"]
            arrays.append(
                (
                    frame["Frequency_Hz"].to_numpy(),
                    frame["Zre_exp_Ohm"].to_numpy() + 1j * frame["Zim_exp_Ohm"].to_numpy(),
                )
            )
        overlap = None
        if all(len(f) >= 2 and len(np.unique(f)) == len(f) for f, _ in arrays):
            overlap = (max(np.min(f) for f, _ in arrays), min(np.max(f) for f, _ in arrays))
            if overlap[0] >= overlap[1]:
                overlap = None
        if overlap is None:
            mismatches.append("no_unambiguous_shared_frequency_window")
        status = "incompatible" if mismatches else "conditional" if missing else "comparable"
        measured = {}
        if status == "comparable":
            for query in config.probe_frequencies_hz:
                samples = [interpolate_impedance(f, z, query) for f, z in arrays]
                delta = samples[1] - samples[0] if all(z is not None for z in samples) else None
                for part in ("real", "imag"):
                    measured[f"delta_z_{part}_at_{query:.15g}_hz"] = metric(
                        getattr(delta, part) if delta is not None else None,
                        "Ohm",
                        reason=None
                        if delta is not None
                        else "frequency_not_bracketed_by_both_spectra",
                    )
        fits = [
            next(r for r in records if r["record_id"] == f"response.eis_ecm:{s['spectrum_uid']}")
            for s in pair
        ]
        consistency = [
            next(r for r in records if r["record_id"] == f"response.eis_kk:{s['spectrum_uid']}")
            for s in pair
        ]
        quality_flags = []
        if any(
            r["applicability"]["status"] != "applicable" or r["quality"]["status"] != "ok"
            for r in consistency
        ):
            quality_flags.append("measurement_consistency_not_established")
        parameter_status = "not_computable"
        if status == "comparable" and all(
            r["applicability"]["status"] == "applicable" for r in fits
        ):
            parameter_status = "conditional"
            for name, value in fits[0]["metrics"].items():
                if name in {"rmse", "nrmse"}:
                    continue
                other = fits[1]["metrics"].get(name)
                if other is not None and value["unit"] == other["unit"]:
                    if name.startswith("Q"):
                        exponent = {"Q1": "n1", "Q2": "n2", "Qd": "nd"}[name]
                        if (
                            fits[0]["metrics"][exponent]["value"]
                            != fits[1]["metrics"][exponent]["value"]
                        ):
                            measured[f"delta_model_parameter_{name}"] = metric(
                                None, value["unit"], reason="cpe_exponent_dependent_units_differ"
                            )
                            continue
                    measured[f"delta_model_parameter_{name}"] = metric(
                        other["value"] - value["value"], value["unit"]
                    )
        uid = f"{left['spectrum_uid']}:{right['spectrum_uid']}"
        result.append(
            make_record(
                record_id=f"evolution.eis_comparison:{uid}",
                record_type="evolution.eis_comparison",
                cell_id=left["conditions"]["cell_id"]["value"] or "unknown",
                cycle_scope=None,
                source_intervals=[refs[s["spectrum_uid"]] for s in pair],
                attributes={
                    "comparison_status": status,
                    "comparison_kind": config.comparison_kind,
                    "missing_conditions": missing,
                    "mismatched_conditions": mismatches,
                    "conditions": {s["spectrum_uid"]: s["conditions"] for s in pair},
                    "ecm_parameter_comparison_status": parameter_status,
                    "ecm_parameter_comparison_limit": "parameter_stability_and_physical_arc_identity_not_assessed",
                    "common_frequency_window_hz": list(overlap) if overlap else None,
                    "supporting_record_ids": [
                        r["record_id"]
                        for r in records
                        if r["record_id"].endswith(tuple(s["spectrum_uid"] for s in pair))
                    ],
                },
                metrics=measured,
                provider="BFL",
                method_name="condition_gated_spectral_comparison_v1",
                provider_version="BFL 0.4.0",
                parameters={
                    "tolerances": config.comparison_tolerances,
                    "interpolation": "linear_real_imag_in_log10_frequency",
                },
                applicability_status={
                    "comparable": "matched",
                    "conditional": "partial",
                    "incompatible": "unmatched",
                }[status],
                applicability_reasons=[*missing, *mismatches],
                quality_status="ok" if status == "comparable" and not quality_flags else "warning",
                quality_flags=quality_flags,
                interpretation_limits=[
                    "Differences describe measurements under the recorded comparison conditions.",
                    "A comparison does not establish ageing causality or a degradation mechanism.",
                    "ECM arc identity and parameter stability are not established by canonical numbering.",
                ],
            )
        )
    return result
