"""Single public API for Battery Feature Lab."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

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
