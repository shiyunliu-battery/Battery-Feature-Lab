"""End-to-end battery feature extraction pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from battery_feature_lab.analysis.degradation_tags import build_degradation_tags
from battery_feature_lab.bds_adapter.readers import read_bds_export_with_report
from battery_feature_lab.evidence import (
    build_evidence_candidates,
    score_evidence_candidates,
    select_evidence,
)
from battery_feature_lab.evidence.jsonl_writer import write_evidence_jsonl
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
from battery_feature_lab.protocol import (
    annotate_normalized,
    build_protocol_records,
    detect_protocol_segments,
    write_protocol_jsonl,
)
from battery_feature_lab.schemas import (
    DiagnosticConfig,
    EvidenceConfig,
    ExportConfig,
    FeatureConfig,
    ProtocolConfig,
    ReaderConfig,
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the end-to-end feature extraction pipeline."""

    reader: ReaderConfig
    features: FeatureConfig
    export: ExportConfig
    diagnostics: DiagnosticConfig = DiagnosticConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    protocol: ProtocolConfig = ProtocolConfig()


class FeaturePipeline:
    """Orchestrate import, feature extraction, domain tagging and export."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._bds_report: dict = {}
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

        normalized, bds_report = read_bds_export_with_report(input_path, self.config.reader)
        self._bds_report = bds_report
        tables: dict[str, pd.DataFrame] = {}

        if self.config.protocol.enabled:
            protocol_segments = detect_protocol_segments(
                normalized,
                self.config.protocol,
                nominal_capacity_ah=self.config.features.nominal_capacity_ah,
            )
            normalized = annotate_normalized(normalized, protocol_segments)
            tables["protocol_segments"] = protocol_segments

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

        if self.config.evidence.enabled:
            candidates = build_evidence_candidates(tables, tags)
            scored_candidates = score_evidence_candidates(candidates, self.config.evidence)
            selected_evidence = select_evidence(scored_candidates, self.config.evidence)
            tables["evidence_candidates"] = scored_candidates
            tables["selected_evidence"] = selected_evidence

        written = write_feature_tables(
            tables,
            self.config.export.output_dir,
            compression=self.config.export.parquet_compression,
        )
        if self.config.evidence.enabled:
            write_evidence_jsonl(
                tables.get("evidence_candidates"),
                self.config.export.output_dir / "evidence_candidates.jsonl",
            )
            write_evidence_jsonl(
                tables.get("selected_evidence"),
                self.config.export.output_dir / "selected_evidence.jsonl",
            )
        if self.config.protocol.enabled:
            write_protocol_jsonl(
                build_protocol_records(tables.get("protocol_segments")),
                self.config.export.output_dir / "protocol_segments.jsonl",
            )
        llm_records = build_llm_context_records(
            {
                name: frame
                for name, frame in tables.items()
                if name not in {
                    "degradation_tags",
                    "evidence_candidates",
                    "selected_evidence",
                    "protocol_segments",
                }
            },
            tags,
            metadata=self._llm_metadata(),
        )
        write_llm_jsonl(llm_records, self.config.export.output_dir / "llm_context.jsonl")
        self._write_metadata(input_path, written, llm_record_count=len(llm_records))
        return tables

    def _llm_metadata(self) -> dict:
        return {
            "cell_context": {
                "nominal_capacity_ah": self.config.features.nominal_capacity_ah,
                "chemistry": None,
            },
            "analysis_config": {
                "reader_config": {
                    "ingest": "battery-data-standard",
                    "current_rest_threshold_a": self.config.reader.current_rest_threshold_a,
                    "soc_unit": self.config.reader.soc_unit,
                },
                "feature_config": asdict(self.config.features),
                "diagnostic_config": asdict(self.config.diagnostics),
                "evidence_config": asdict(self.config.evidence),
                "protocol_config": asdict(self.config.protocol),
            },
        }

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
            "bds_conversion_report": self._bds_report,
            "feature_config": self.config.features.__dict__,
            "diagnostic_config": self.config.diagnostics.__dict__,
            "evidence_config": self.config.evidence.__dict__,
            "protocol_config": self.config.protocol.__dict__,
            "reader_config": {
                "cell_id": self.config.reader.cell_id,
                "ingest": "battery-data-standard",
                "current_rest_threshold_a": self.config.reader.current_rest_threshold_a,
                "soc_unit": self.config.reader.soc_unit,
            },
        }
        path = self.config.export.output_dir / "run_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
