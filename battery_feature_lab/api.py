"""Public time-series and spectrum analysis entry points."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from battery_feature_lab.analysis.compiler import compile_analysis
from battery_feature_lab.analysis.schema import AnalysisConfig, AnalysisResult


def analyze(
    input_path: str | Path,
    output_dir: str | Path = "bfl_outputs",
    *,
    input_adapter: str = "auto",
    cell_id: str | None = None,
    nominal_capacity_ah: float | None = None,
    representative_cycle: int | None = None,
    declared_protocol_name: str | None = None,
    formation_cycles_to_exclude: int = 1,
    reference_window_size: int = 4,
    pulse_resistance_times_s: tuple[float, ...] = (10.0,),
    relaxation_checkpoints_s: tuple[float, ...] = (10.0, 30.0, 60.0, 300.0, 600.0, 1800.0),
    voltage_column: str | None = None,
    temperature_column: str | None = None,
    analysis_policy: Mapping[str, float] | None = None,
) -> AnalysisResult:
    """Analyze a supported cycler file and write the machine-readable BFL contract."""

    config = AnalysisConfig(
        output_dir=Path(output_dir),
        input_adapter=input_adapter,
        cell_id=cell_id,
        nominal_capacity_ah=nominal_capacity_ah,
        representative_cycle=representative_cycle,
        declared_protocol_name=declared_protocol_name,
        formation_cycles_to_exclude=formation_cycles_to_exclude,
        reference_window_size=reference_window_size,
        pulse_resistance_times_s=pulse_resistance_times_s,
        relaxation_checkpoints_s=relaxation_checkpoints_s,
        voltage_column=voltage_column,
        temperature_column=temperature_column,
        analysis_policy=dict(analysis_policy or {}),
    )
    return compile_analysis(Path(input_path), config)


def analyze_eis(
    input_path: str | Path,
    output_dir: str | Path = "bfl_eis_outputs",
    *,
    sheet: str | int | None = None,
    single_spectrum: bool = False,
    group_columns: tuple[str, ...] = (),
    columns: Mapping[str, str] | None = None,
    phase_unit: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    metadata_path: str | Path | None = None,
    probe_frequencies_hz: tuple[float, ...] = (1.0, 10.0, 1000.0),
    run_kk: bool = True,
    run_fit: bool = True,
    fit_options: Mapping[str, Any] | None = None,
    kk_options: Mapping[str, Any] | None = None,
    minimum_provider_points: int = 12,
    kk_residual_limit: float = 0.05,
    fit_nrmse_limit: float = 0.05,
    comparison_pairs: tuple[tuple[str, str], ...] = (),
    comparison_kind: str = "repeat",
    comparison_tolerances: Mapping[str, float] | None = None,
) -> AnalysisResult:
    """Analyze independently identified EIS spectra and write six evidence artifacts."""
    from battery_feature_lab.eis.compiler import compile_eis
    from battery_feature_lab.eis.schema import EISConfig

    metadata_artifact = {}
    if metadata_path is not None:
        from battery_feature_lab.analysis.writer import sha256_file

        if metadata is not None:
            raise ValueError("Supply metadata or metadata_path, not both")
        metadata_file = Path(metadata_path)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata_artifact = {"filename": metadata_file.name, "sha256": sha256_file(metadata_file)}

    config = EISConfig(
        output_dir=Path(output_dir), sheet=sheet, single_spectrum=single_spectrum,
        group_columns=tuple(group_columns), columns=dict(columns or {}), phase_unit=phase_unit,
        metadata=dict(metadata or {}), metadata_artifact=metadata_artifact,
        probe_frequencies_hz=tuple(probe_frequencies_hz),
        run_kk=run_kk, run_fit=run_fit, fit_options=dict(fit_options or {}),
        kk_options=dict(kk_options or {}), minimum_provider_points=minimum_provider_points,
        kk_residual_limit=kk_residual_limit, fit_nrmse_limit=fit_nrmse_limit,
        comparison_pairs=tuple(comparison_pairs), comparison_kind=comparison_kind,
        comparison_tolerances=dict(comparison_tolerances or {}),
    )
    return compile_eis(Path(input_path), config)
