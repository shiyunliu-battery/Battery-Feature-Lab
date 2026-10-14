"""Observed spectra and separately audited Lin-KK and circuit-fit results."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from battery_feature_lab._version import PROVIDER_VERSION
from battery_feature_lab.analysis.provider_output import capture_provider_output
from battery_feature_lab.analysis.schema import make_record, metric
from battery_feature_lab.eis.schema import EISConfig

LIMITS = [
    "EIS descriptors depend on measurement conditions and frequency coverage.",
    "A spectrum or circuit parameter alone does not establish a degradation mechanism.",
]
EISFIT_REPOSITORY = "https://github.com/shiyunliu-battery/Battery-EIS-Fit"


def eisfit_reference() -> str:
    """Return the release page of the installed eisfit, or its repository if absent."""
    try:
        return f"{EISFIT_REPOSITORY}/releases/tag/v{version('eisfit')}"
    except PackageNotFoundError:
        return EISFIT_REPOSITORY


class InvalidProviderResult(ValueError):
    """A provider returned values that cannot be serialized as a valid result."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def value_class(value: float) -> str:
    """Name the IEEE class of a provider scalar without coercing it to JSON."""
    if np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "positive_infinity" if value > 0 else "negative_infinity"
    return "finite"


def interpolate_impedance(f, z, query):
    """Interpolate real and imaginary parts in log-frequency; never extrapolate."""
    if len(f) < 2 or query < np.min(f) or query > np.max(f) or len(np.unique(f)) != len(f):
        return None
    order = np.argsort(f)
    x = np.log10(f[order])
    return complex(
        np.interp(np.log10(query), x, z.real[order]), np.interp(np.log10(query), x, z.imag[order])
    )


def record(spectrum, kind, source_ref, **kwargs):
    return make_record(
        record_id=f"{kind}:{spectrum['spectrum_uid']}",
        record_type=kind,
        cell_id=spectrum["conditions"]["cell_id"]["value"] or "unknown",
        cycle_scope=None,
        source_intervals=[source_ref],
        provider_version=kwargs.pop("provider_version", PROVIDER_VERSION),
        interpretation_limits=kwargs.pop("interpretation_limits", LIMITS),
        **kwargs,
    )


def analyze_spectrum(spectrum, source_ref, config: EISConfig, calls):
    frame = spectrum["frame"]
    f = frame["Frequency_Hz"].to_numpy()
    z = frame["Zre_exp_Ohm"].to_numpy() + 1j * frame["Zim_exp_Ohm"].to_numpy()
    uid = spectrum["spectrum_uid"]
    duplicates = len(f) - len(np.unique(f))
    flags = ["duplicate_frequencies"] if duplicates else []
    excluded = len(spectrum["source_rows"]) - len(f)
    if excluded:
        flags.append("invalid_points_excluded")
    metrics = {
        "valid_point_count": metric(len(f), "1"),
        "excluded_point_count": metric(excluded, "1"),
        "duplicate_frequency_count": metric(duplicates, "1"),
        "frequency_min": metric(float(np.min(f)) if len(f) else None, "Hz"),
        "frequency_max": metric(float(np.max(f)) if len(f) else None, "Hz"),
    }
    for name, values in (("real", z.real), ("imaginary", z.imag), ("modulus", np.abs(z))):
        metrics[f"impedance_{name}_min"] = metric(float(np.min(values)) if len(f) else None, "Ohm")
        metrics[f"impedance_{name}_max"] = metric(float(np.max(values)) if len(f) else None, "Ohm")
    for query in config.probe_frequencies_hz:
        sample = interpolate_impedance(f, z, query)
        for part in ("real", "imag"):
            metrics[f"z_{part}_at_{query:.15g}_hz"] = metric(
                getattr(sample, part) if sample is not None else None,
                "Ohm",
                reason=None if sample is not None else "frequency_not_bracketed_or_duplicate_grid",
            )
    records = [
        record(
            spectrum,
            "operation.eis_conditions",
            source_ref,
            provider="BFL",
            method_name="declared_eis_measurement_context_v1",
            attributes={
                "spectrum_id": spectrum["spectrum_id"],
                "spectrum_uid": uid,
                "conditions": spectrum["conditions"],
            },
        ),
        record(
            spectrum,
            "response.eis_spectrum",
            source_ref,
            provider="BDS+BFL",
            method_name="observed_eis_spectrum_v1",
            attributes={
                "measurement_qc": "warning" if flags or not len(f) else "valid_numeric_points"
            },
            metrics=metrics,
            series={
                "frequency_hz": f.tolist(),
                "zreal_ohm": z.real.tolist(),
                "zimag_ohm": z.imag.tolist(),
                "source_rows": frame["measurement_source_row"].to_list(),
            },
            parameters={
                "interpolation": "linear_real_imag_in_log10_frequency",
                "extrapolation": False,
            },
            quality_status="warning" if flags or not len(f) else "ok",
            quality_flags=flags + (["no_valid_points"] if not len(f) else []),
            applicability_status="applicable" if len(f) else "not_computable",
            applicability_reasons=[] if len(f) else ["no_valid_points"],
        ),
    ]
    ineligible = (
        "duplicate_frequencies"
        if duplicates
        else (
            "insufficient_unique_frequency_points"
            if len(f) < config.minimum_provider_points
            else None
        )
    )
    for method, enabled in (("kk", config.run_kk), ("ecm", config.run_fit)):
        provider = "impedance.py" if method == "kk" else "eisfit"
        parameters = (
            {"residual_limit": config.kk_residual_limit, **config.kk_options}
            if method == "kk"
            else {"fit_nrmse_limit": config.fit_nrmse_limit, **config.fit_options}
        )
        reason = "disabled_by_configuration" if not enabled else ineligible
        call = {
            "spectrum_uid": uid,
            "provider": provider,
            "method": method,
            "status": "not_invoked" if reason else "ok",
        }
        attributes: dict[str, Any] = (
            {"measurement_consistency": "not_assessed"}
            if method == "kk"
            else {
                "convergence": "not_assessed",
                "fit_quality": "not_assessed",
                "parameter_stability": "not_assessed",
                "uncertainty": "not_assessed",
                "identifiability": "not_assessed",
            }
        )
        measured, series, quality_flags = {}, {}, []
        provider_version = "not_invoked"
        if not reason:
            try:
                if method == "kk":
                    from impedance.validation import linKK

                    provider_version = version("impedance")
                    options = {
                        "c": 0.85,
                        "max_M": 50,
                        "fit_type": "complex",
                        "add_cap": False,
                        **config.kk_options,
                    }
                    parameters.update(options)
                    with capture_provider_output() as provider_output:
                        call["provider_output"] = provider_output
                        count, mu, simulated, residual_real, residual_imag = linKK(
                            f, z, **options
                        )
                    outputs = {
                        "fitted_real": simulated.real,
                        "fitted_imaginary": simulated.imag,
                        "residual_real": residual_real,
                        "residual_imaginary": residual_imag,
                    }
                    invalid_outputs = sorted(
                        name
                        for name, values in outputs.items()
                        if np.shape(values) != f.shape or not np.all(np.isfinite(values))
                    )
                    if invalid_outputs or not np.isfinite(mu):
                        attributes["lin_kk_termination"] = {
                            "rc_element_count": int(count),
                            "mu": float(mu) if np.isfinite(mu) else None,
                            "mu_class": value_class(float(mu)),
                            "invalid_outputs": invalid_outputs,
                        }
                        if not np.isfinite(mu):
                            raise InvalidProviderResult(
                                "lin_kk_mu_not_finite", "Lin-KK returned a non-finite mu"
                            )
                        raise InvalidProviderResult(
                            "lin_kk_invalid_outputs",
                            "Lin-KK returned non-finite or misshaped arrays",
                        )
                    magnitude = np.maximum(np.abs(residual_real), np.abs(residual_imag))
                    maximum = float(np.max(magnitude))
                    above = int(np.sum(magnitude > config.kk_residual_limit))
                    attributes["measurement_consistency"] = (
                        "within_configured_residual_limit"
                        if maximum <= config.kk_residual_limit
                        else "outside_configured_residual_limit"
                    )
                    measured = {
                        "maximum_normalized_residual": metric(maximum, "1"),
                        "frequency_of_maximum_residual": metric(
                            float(f[int(np.argmax(magnitude))]), "Hz"
                        ),
                        "points_above_residual_limit": metric(above, "1"),
                        "fraction_points_above_residual_limit": metric(above / len(f), "1"),
                        "mu": metric(float(mu), "1"),
                        "rc_element_count": metric(int(count), "1"),
                    }
                    series = {
                        "frequency_hz": f.tolist(),
                        "residual_real": residual_real.tolist(),
                        "residual_imaginary": residual_imag.tolist(),
                    }
                    if maximum > config.kk_residual_limit:
                        quality_flags.append("lin_kk_residual_limit_exceeded")
                else:
                    import eisfit

                    provider_version = eisfit.__version__
                    options = eisfit.FitConfig(**config.fit_options)
                    parameters.update(asdict(options))
                    with capture_provider_output() as provider_output:
                        call["provider_output"] = provider_output
                        fit = eisfit.fit_model_eis(f, z, options)
                    if not fit.optimizer.success or not np.isfinite(fit.nrmse):
                        raise ValueError("ECM provider returned an unconverged or invalid fit")
                    if fit.Zsim.shape != z.shape or not np.all(np.isfinite(fit.Zsim)):
                        raise ValueError("ECM provider returned an invalid fitted spectrum")
                    undeclared = sorted(set(fit.parameters) - set(fit.parameter_units))
                    if undeclared:
                        attributes.update(convergence="converged", undeclared_parameters=undeclared)
                        raise InvalidProviderResult(
                            "ecm_parameter_units_undeclared",
                            f"eisfit returned parameters without a declared unit: {undeclared}",
                        )
                    diagnostics = fit.diagnostics
                    attributes.update(
                        {
                            "convergence": "converged",
                            "diagnostics": diagnostics,
                            "multistart_near_best": diagnostics["near_best"],
                            "fit_quality": "within_configured_limit"
                            if fit.nrmse <= config.fit_nrmse_limit
                            else "outside_configured_limit",
                            "circuit": "L + R0 + (R1 || CPE1) + (R2 || CPE2) + " + options.tail,
                            "parameter_units": {
                                name: fit.parameter_units[name] for name in fit.parameters
                            },
                        }
                    )
                    measured = {
                        name: metric(value, attributes["parameter_units"][name])
                        for name, value in fit.parameters.items()
                    }
                    measured.update(
                        {"rmse": metric(fit.rmse, "Ohm"), "nrmse": metric(fit.nrmse, "1")}
                    )
                    series = {
                        "frequency_hz": f.tolist(),
                        "fitted_zreal_ohm": fit.Zsim.real.tolist(),
                        "fitted_zimag_ohm": fit.Zsim.imag.tolist(),
                        "residual_zreal_ohm": (z - fit.Zsim).real.tolist(),
                        "residual_zimag_ohm": (z - fit.Zsim).imag.tolist(),
                    }
                    quality_flags = (
                        ["fit_nrmse_limit_exceeded"] if fit.nrmse > config.fit_nrmse_limit else []
                    )
                    if fit.bound_hits:
                        quality_flags.append("parameter_bound_hits")
                    kk = records[-1]
                    if kk["record_type"] == "response.eis_kk" and (
                        kk["quality"]["status"] != "ok"
                        or kk["applicability"]["status"] != "applicable"
                    ):
                        quality_flags.append("measurement_consistency_not_established")
            except ImportError as exc:
                reason = "provider_unavailable; install battery-feature-lab[eis]"
                call.update(status="error", error=str(exc))
            except InvalidProviderResult as exc:
                reason = f"provider_invalid_result:{exc.code}"
                call.update(status="error", error=str(exc))
            except Exception as exc:  # noqa: BLE001 - provider failures are retained as evidence
                reason = f"provider_error:{type(exc).__name__}:{exc}"
                call.update(status="error", error=str(exc))
                if hasattr(exc, "attempts"):
                    attributes.update(
                        convergence="not_converged", attempts=[asdict(a) for a in exc.attempts]
                    )
            if reason:
                measured, series = {}, {}
        if reason:
            call["reason"] = reason
        call["version"] = provider_version
        calls.append(call)
        records.append(
            record(
                spectrum,
                f"response.eis_{method}",
                source_ref,
                provider=provider,
                method_name="lin_kk_v1" if method == "kk" else "two_arc_ecm_v1",
                provider_version=provider_version,
                attributes=attributes,
                metrics=measured,
                series=series,
                parameters=parameters,
                references=["https://doi.org/10.1016/j.electacta.2014.01.034"]
                if method == "kk"
                else [eisfit_reference()],
                applicability_status="not_computable"
                if reason
                else "partial"
                if quality_flags
                else "applicable",
                applicability_reasons=[reason] if reason else [],
                quality_status="warning" if reason or quality_flags else "ok",
                quality_flags=quality_flags,
                interpretation_limits=LIMITS
                + [
                    "Lin-KK consistency is conditional on coverage and configuration; it does not prove a circuit model.",
                    "Arc numbering is a time-order convention, not persistent physical-process identity.",
                    "Convergence and fit error do not establish parameter identifiability or uncertainty.",
                ],
            )
        )
    return records
