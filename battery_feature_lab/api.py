"""User-facing convenience API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from battery_feature_lab.pipeline import FeaturePipeline, PipelineConfig
from battery_feature_lab.schemas import DiagnosticConfig, ExportConfig, FeatureConfig, ReaderConfig


@dataclass(frozen=True)
class ExtractionResult:
    """Result returned by :func:`extract`."""

    tables: dict[str, pd.DataFrame]
    output_dir: Path

    @property
    def files(self) -> list[Path]:
        """Files written to the output directory."""

        return sorted(self.output_dir.glob("*"))

    @property
    def llm_context_path(self) -> Path:
        """Path to the JSONL context summary."""

        return self.output_dir / "llm_context.jsonl"

    @property
    def metadata_path(self) -> Path:
        """Path to the run metadata JSON output."""

        return self.output_dir / "run_metadata.json"


def extract(
    input_path: str | Path,
    output_dir: str | Path = "bfl_outputs",
    *,
    cell_id: str | None = None,
    nominal_capacity_ah: float | None = None,
    positive_current_is_charge: bool = True,
    reference_cycle: int = 10,
    target_cycle: int = 100,
    high_soc_rest_threshold: float | None = 0.5,
    datasheet_max_discharge_c_rate: float | None = None,
    write_normalized_timeseries: bool = True,
    **feature_overrides: Any,
) -> ExtractionResult:
    """Extract battery features from a BDS/cycler export with minimal setup.

    Parameters mirror the most common CLI options. Use the lower-level ``FeaturePipeline`` API
    when you need complete control over every configuration object.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    feature_kwargs = {
        "nominal_capacity_ah": nominal_capacity_ah,
        "early_reference_cycle": reference_cycle,
        "early_target_cycle": target_cycle,
        **feature_overrides,
    }
    config = PipelineConfig(
        reader=ReaderConfig(
            cell_id=cell_id or input_path.stem,
            positive_current_is_charge=positive_current_is_charge,
        ),
        features=FeatureConfig(**feature_kwargs),
        diagnostics=DiagnosticConfig(
            high_soc_rest_fraction_threshold=high_soc_rest_threshold,
            datasheet_max_discharge_c_rate=datasheet_max_discharge_c_rate,
        ),
        export=ExportConfig(
            output_dir=output_dir,
            write_normalized_timeseries=write_normalized_timeseries,
        ),
    )
    tables = FeaturePipeline(config).run(input_path)
    return ExtractionResult(tables=tables, output_dir=output_dir)
