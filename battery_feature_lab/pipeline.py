"""End-to-end battery feature extraction pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from battery_feature_lab.analysis.degradation_tags import build_degradation_tags
from battery_feature_lab.bds_adapter.readers import read_bds_export
from battery_feature_lab.export.llm_json_writer import build_llm_context_records, write_llm_jsonl
from battery_feature_lab.export.parquet_writer import write_feature_tables
from battery_feature_lab.featurizers import (
    CycleSummaryFeaturizer,
    DeltaQFeaturizer,
    EISDRTFeaturizer,
    ICADVAFeaturizer,
    RelaxationFeaturizer,
    StressHistogramFeaturizer,
)
from battery_feature_lab.schemas import DiagnosticConfig, ExportConfig, FeatureConfig, ReaderConfig


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the end-to-end feature extraction pipeline."""

    reader: ReaderConfig
    features: FeatureConfig
    export: ExportConfig
    diagnostics: DiagnosticConfig = DiagnosticConfig()


class FeaturePipeline:
    """Orchestrate import, feature extraction, domain tagging and export."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.featurizers = [
            CycleSummaryFeaturizer(config.features),
            DeltaQFeaturizer(config.features),
            ICADVAFeaturizer(config.features),
            RelaxationFeaturizer(config.features),
            StressHistogramFeaturizer(config.features),
            EISDRTFeaturizer(config.features),
        ]

    def run(self, input_path: str | Path) -> dict[str, pd.DataFrame]:
        """Run the full pipeline and write outputs."""

        normalized = read_bds_export(input_path, self.config.reader)
        tables: dict[str, pd.DataFrame] = {}
        if self.config.export.write_normalized_timeseries:
            tables["normalized_timeseries"] = normalized

        for featurizer in self.featurizers:
            table = featurizer.extract(normalized)
            tables[table.name] = table.frame

        tags = build_degradation_tags(
            cycle_features=tables.get("cycle_features"),
            ica_dva_features=tables.get("ica_dva_features"),
            stress_features=tables.get("stress_features"),
            relaxation_features=tables.get("relaxation_features"),
            config=self.config.diagnostics,
        )
        tables["degradation_tags"] = tags

        written = write_feature_tables(
            tables,
            self.config.export.output_dir,
            compression=self.config.export.parquet_compression,
        )
        llm_records = build_llm_context_records(
            {name: frame for name, frame in tables.items() if name != "degradation_tags"},
            tags,
        )
        write_llm_jsonl(llm_records, self.config.export.output_dir / "llm_context.jsonl")
        self._write_metadata(input_path, written, llm_record_count=len(llm_records))
        return tables

    def _write_metadata(
        self,
        input_path: str | Path,
        written: dict[str, Path],
        *,
        llm_record_count: int,
    ) -> None:
        metadata = {
            "input_path": str(input_path),
            "output_dir": str(self.config.export.output_dir),
            "written_tables": {name: str(path) for name, path in written.items()},
            "llm_record_count": llm_record_count,
            "feature_config": self.config.features.__dict__,
            "diagnostic_config": self.config.diagnostics.__dict__,
            "reader_config": {
                "cell_id": self.config.reader.cell_id,
                "positive_current_is_charge": self.config.reader.positive_current_is_charge,
                "current_rest_threshold_a": self.config.reader.current_rest_threshold_a,
                "time_unit": self.config.reader.time_unit,
                "capacity_unit": self.config.reader.capacity_unit,
                "soc_unit": self.config.reader.soc_unit,
            },
        }
        path = self.config.export.output_dir / "run_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
