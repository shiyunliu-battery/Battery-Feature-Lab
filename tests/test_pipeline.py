from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from battery_feature_lab.analysis.degradation_tags import build_degradation_tags, mann_kendall_sen_slope
from battery_feature_lab.pipeline import FeaturePipeline, PipelineConfig
from battery_feature_lab.schemas import DiagnosticConfig, ExportConfig, FeatureConfig, ReaderConfig


def make_synthetic_bds(cycles: int = 20, *, include_soc: bool = False) -> pd.DataFrame:
    rows = []; nominal = 1.1; t = 0.0
    for cycle in range(1, cycles + 1):
        qmax = nominal * (1.0 - 0.0008 * cycle)
        for step, current, count in ((0, 1.1, 40), (1, 0.0, 12), (2, -1.1, 45)):
            for k in range(count):
                frac = k / max(count - 1, 1)
                voltage = 3.0 + 0.55 * frac if step == 0 else (3.55 - 0.04 * frac if step == 1 else 3.45 - 0.6 * frac)
                row = {"cell": "synthetic_cell", "cycle": cycle, "step": step, "time": t, "voltage": voltage, "current": current, "temperature": 25.0, "charge_capacity": qmax * frac if step == 0 else qmax, "discharge_capacity": qmax * frac if step == 2 else 0.0}
                if include_soc:
                    row["soc"] = frac if step == 0 else (1.0 if step == 1 else 1.0 - frac)
                rows.append(row); t += 30
    return pd.DataFrame(rows)


def _run(tmp_path: Path, *, include_soc: bool = False):
    path = tmp_path / "cell.csv"; make_synthetic_bds(include_soc=include_soc).to_csv(path, index=False)
    return FeaturePipeline(PipelineConfig(reader=ReaderConfig(cell_id="synthetic_cell"), features=FeatureConfig(nominal_capacity_ah=1.1, early_reference_cycle=2, early_target_cycle=10), export=ExportConfig(output_dir=tmp_path / "out"))).run(path)


def test_llm_context_is_evidence_backed(tmp_path: Path) -> None:
    tables = _run(tmp_path)
    assert not tables["evidence_candidates"].empty
    context = json.loads((tmp_path / "out" / "llm_context.jsonl").read_text().splitlines()[0])
    assert context["schema_version"] == "3.0"
    assert context["evidence_chain"]["feature_table_bypass"] is False
    assert context["evidence"] and context["contracts"]
    assert (tmp_path / "out" / "feature_contracts.json").exists()


def test_missing_soc_stays_not_computable(tmp_path: Path) -> None:
    tables = _run(tmp_path, include_soc=False)
    stress = tables["stress_features"].iloc[0]
    assert stress["soc_source"] == "unavailable"
    assert pd.isna(stress["high_soc_rest_fraction"])
    assert "high_soc_rest_fraction" not in set(tables["evidence_candidates"]["feature_name"])


def test_explicit_soc_is_used(tmp_path: Path) -> None:
    tables = _run(tmp_path, include_soc=True)
    stress = tables["stress_features"].iloc[0]
    assert stress["soc_source"] == "provided"
    assert pd.notna(stress["high_soc_rest_fraction"])


def test_degradation_output_is_observation_only() -> None:
    cycles = pd.DataFrame({"cell_id": ["A"] * 20, "cycle_index": np.arange(1, 21), "discharge_capacity_ah": 1.0 - 0.01 * np.arange(20)})
    tags = build_degradation_tags(cycle_features=cycles, config=DiagnosticConfig())
    assert "capacity_decreasing_trend" in set(tags["signal"])
    assert "possible_modes" not in tags.columns
    assert set(tags["kind"]) == {"observation"}


def test_mann_kendall_sen_slope_detects_monotonic_fade() -> None:
    trend = mann_kendall_sen_slope(np.arange(20), 1.0 - 0.01 * np.arange(20))
    assert trend["p_value"] < 0.05
    assert trend["sen_slope"] < 0
