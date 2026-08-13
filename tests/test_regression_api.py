from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import bfl

from battery_feature_lab.cli import main


def _small_dataset(cycles: int = 15) -> pd.DataFrame:
    rows = []
    t = 0.0
    nominal = 1.1
    for cycle in range(1, cycles + 1):
        qmax = nominal * (1.0 - 0.0008 * cycle)
        for step, current, count in ((0, 1.1, 30), (1, 0.0, 8), (2, -1.1, 35)):
            for index in range(count):
                fraction = index / max(count - 1, 1)
                voltage = 3.0 + 0.55 * fraction if step == 0 else (3.55 - 0.03 * fraction if step == 1 else 3.45 - 0.6 * fraction)
                rows.append({"cell": "synthetic_cell", "cycle": cycle, "step": step, "time": t, "voltage": voltage, "current": current, "temperature": 25.0, "charge_capacity": qmax * fraction if step == 0 else qmax, "discharge_capacity": qmax * fraction if step == 2 else 0.0})
                t += 30
    return pd.DataFrame(rows)


def test_cli_writes_existing_outputs_and_contract_catalogue(tmp_path: Path) -> None:
    path = tmp_path / "bds.csv"
    _small_dataset().to_csv(path, index=False)
    exit_code = main(["extract", str(path), "--output-dir", str(tmp_path / "cli_out"), "--cell-id", "synthetic_cell", "--nominal-capacity-ah", "1.1", "--reference-cycle", "2", "--target-cycle", "10", "--evidence-question", "Why did capacity fade?", "--evidence-token-budget", "240"])
    assert exit_code == 0
    assert (tmp_path / "cli_out" / "cycle_features.parquet").exists()
    assert (tmp_path / "cli_out" / "selected_evidence.jsonl").exists()
    assert (tmp_path / "cli_out" / "feature_contracts.json").exists()
    records = [json.loads(line) for line in (tmp_path / "cli_out" / "selected_evidence.jsonl").read_text().splitlines()]
    assert records
    assert sum(record["token_cost"] for record in records) <= 240


def test_short_bfl_api_still_extracts_features(tmp_path: Path) -> None:
    path = tmp_path / "bds.csv"
    _small_dataset().to_csv(path, index=False)
    result = bfl.extract(path, output_dir=tmp_path / "short_api_out", nominal_capacity_ah=1.1, reference_cycle=2, target_cycle=10)
    assert not result.tables["cycle_features"].empty
    assert not result.tables["selected_evidence"].empty
    assert result.llm_context_path.exists()
    assert result.selected_evidence_path.exists()
    assert result.metadata_path.exists()
    assert any(output.name == "cycle_features.parquet" for output in result.files)
