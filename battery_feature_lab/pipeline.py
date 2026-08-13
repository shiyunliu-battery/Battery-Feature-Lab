"""End-to-end battery feature extraction pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from battery_feature_lab.analysis.degradation_tags import build_degradation_tags
from battery_feature_lab.bds_adapter.readers import read_bds_export_with_report
from battery_feature_lab.evidence import build_evidence_candidates, score_evidence_candidates, select_evidence
from battery_feature_lab.evidence.jsonl_writer import write_evidence_jsonl
from battery_feature_lab.export.evidence_context_writer import build_llm_context_records, write_llm_jsonl
from battery_feature_lab.export.parquet_writer import write_feature_tables
from battery_feature_lab.feature_contracts import write_feature_contracts_json
from battery_feature_lab.featurizers import CycleSummaryFeaturizer, DeltaQFeaturizer, EISDRTFeaturizer, ICADVAFeaturizer, RelaxationFeaturizer, StressHistogramFeaturizer
from battery_feature_lab.protocol import annotate_normalized, build_protocol_records, detect_protocol_segments, write_protocol_jsonl
from battery_feature_lab.schemas import DiagnosticConfig, EvidenceConfig, ExportConfig, FeatureConfig, ProtocolConfig, ReaderConfig


@dataclass(frozen=True)
class PipelineConfig:
    reader: ReaderConfig
    features: FeatureConfig
    export: ExportConfig
    diagnostics: DiagnosticConfig = DiagnosticConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    protocol: ProtocolConfig = ProtocolConfig()


class FeaturePipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._bds_report: dict = {}
        self.featurizers = [CycleSummaryFeaturizer(config.features), DeltaQFeaturizer(config.features), ICADVAFeaturizer(config.features), RelaxationFeaturizer(config.features), StressHistogramFeaturizer(config.features), EISDRTFeaturizer(config.features)]

    def run(self, input_path: str | Path) -> dict[str, pd.DataFrame]:
        normalized, bds_report = read_bds_export_with_report(input_path, self.config.reader)
        self._bds_report = bds_report
        tables: dict[str, pd.DataFrame] = {}
        if self.config.protocol.enabled:
            protocol_segments = detect_protocol_segments(normalized, self.config.protocol, nominal_capacity_ah=self.config.features.nominal_capacity_ah)
            normalized = annotate_normalized(normalized, protocol_segments)
            tables["protocol_segments"] = protocol_segments
        if self.config.export.write_normalized_timeseries:
            tables["normalized_timeseries"] = normalized
        for featurizer in self.featurizers:
            table = featurizer.extract(normalized)
            tables[table.name] = table.frame
        observations = build_degradation_tags(cycle_features=tables.get("cycle_features"), ica_dva_features=tables.get("ica_dva_features"), stress_features=tables.get("stress_features"), relaxation_features=tables.get("relaxation_features"), config=self.config.diagnostics)
        tables["degradation_tags"] = observations
        contract_parameters = {**asdict(self.config.features), **asdict(self.config.diagnostics), **asdict(self.config.protocol)}
        evidence_input_tables = dict(tables)
        evidence_input_tables["normalized_timeseries"] = normalized
        evidence_candidates = build_evidence_candidates(evidence_input_tables, observations, contract_parameters=contract_parameters)
        context_evidence = evidence_candidates
        if self.config.evidence.enabled:
            scored_candidates = score_evidence_candidates(evidence_candidates, self.config.evidence)
            selected_evidence = select_evidence(scored_candidates, self.config.evidence)
            tables["evidence_candidates"] = scored_candidates
            tables["selected_evidence"] = selected_evidence
            context_evidence = selected_evidence if selected_evidence is not None and not selected_evidence.empty else scored_candidates
        written = write_feature_tables(tables, self.config.export.output_dir, compression=self.config.export.parquet_compression)
        if self.config.evidence.enabled:
            write_evidence_jsonl(tables.get("evidence_candidates"), self.config.export.output_dir / "evidence_candidates.jsonl")
            write_evidence_jsonl(tables.get("selected_evidence"), self.config.export.output_dir / "selected_evidence.jsonl")
        if self.config.protocol.enabled:
            write_protocol_jsonl(build_protocol_records(tables.get("protocol_segments")), self.config.export.output_dir / "protocol_segments.jsonl")
        contracts_path = write_feature_contracts_json(self.config.export.output_dir / "feature_contracts.json")
        llm_records = build_llm_context_records(context_evidence, metadata=self._llm_metadata())
        write_llm_jsonl(llm_records, self.config.export.output_dir / "llm_context.jsonl")
        self._write_metadata(input_path, written, contracts_path=contracts_path, llm_record_count=len(llm_records), llm_evidence_source=("selected_evidence" if self.config.evidence.enabled and tables.get("selected_evidence") is not None and not tables["selected_evidence"].empty else "contracted_evidence_candidates"))
        return tables

    def _llm_metadata(self) -> dict:
        return {"cell_context": {"nominal_capacity_ah": self.config.features.nominal_capacity_ah, "chemistry": None}, "analysis_config": {"reader_config": {"ingest": "battery-data-standard", "current_rest_threshold_a": self.config.reader.current_rest_threshold_a, "soc_unit": self.config.reader.soc_unit}, "feature_config": asdict(self.config.features), "diagnostic_config": asdict(self.config.diagnostics), "evidence_config": asdict(self.config.evidence), "protocol_config": asdict(self.config.protocol)}, "bds_conversion_report": self._bds_report}

    def _write_metadata(self, input_path: str | Path, written: dict[str, Path], *, contracts_path: Path, llm_record_count: int, llm_evidence_source: str) -> None:
        metadata = {"input_path": str(input_path), "output_dir": str(self.config.export.output_dir), "written_tables": {name: str(path) for name, path in written.items()}, "feature_contracts": str(contracts_path), "llm_record_count": llm_record_count, "llm_context_policy": "feature -> Feature Contract -> evidence record -> LLM context", "llm_context_source": llm_evidence_source, "bds_conversion_report": self._bds_report, "feature_config": asdict(self.config.features), "diagnostic_config": asdict(self.config.diagnostics), "evidence_config": asdict(self.config.evidence), "protocol_config": asdict(self.config.protocol), "reader_config": {"cell_id": self.config.reader.cell_id, "ingest": "battery-data-standard", "current_rest_threshold_a": self.config.reader.current_rest_threshold_a, "soc_unit": self.config.reader.soc_unit}}
        path = self.config.export.output_dir / "run_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
