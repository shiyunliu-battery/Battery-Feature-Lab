from __future__ import annotations

import json
import pandas as pd
import pytest

from battery_feature_lab.evidence.candidates import build_evidence_candidates
from battery_feature_lab.evidence.scorer import score_evidence_candidates
from battery_feature_lab.evidence.selector import select_evidence
from battery_feature_lab.schemas import EvidenceConfig


def _candidate(evidence_id: str, text: str, **overrides: object) -> dict[str, object]:
    record = {"cell_id": "cell-1", "evidence_id": evidence_id, "evidence_type": "derived_feature", "source_table": "features", "feature_name": evidence_id, "support_role": "measurement", "text": text, "interpretation_hint": "Observed feature only.", "protocol": "unknown", "token_cost": 20, "redundancy_key": evidence_id}
    record.update(overrides)
    return record


def test_scoring_stays_stable_when_candidate_added() -> None:
    question = "Why did discharge capacity fade?"
    target = _candidate("capacity", "Discharge capacity faded across cycles.")
    first = score_evidence_candidates(pd.DataFrame([target]), EvidenceConfig(question=question))
    expanded = score_evidence_candidates(pd.DataFrame([target, _candidate("temperature", "Mean temperature was stable.")]), EvidenceConfig(question=question))
    target_expanded = expanded.loc[expanded["evidence_id"] == "capacity"].iloc[0]
    assert target_expanded["question_relevance"] == pytest.approx(first.iloc[0]["question_relevance"])


def test_unregistered_numeric_feature_is_not_exposed() -> None:
    table = pd.DataFrame({"cell_id": ["cell-1", "cell-1"], "cycle_index": [1, 10], "new_model_output": [2.0, 3.5]})
    assert build_evidence_candidates({"future_feature_table": table}).empty


def test_contracted_evidence_has_provenance_contract() -> None:
    cycle_features = pd.DataFrame({"cell_id": ["cell-1", "cell-1"], "cycle_index": [1, 10], "discharge_capacity_ah": [2.0, 1.8], "coulombic_efficiency": [0.99, 0.98]})
    normalized = pd.DataFrame({"cell_id": ["cell-1"] * 4, "cycle_index": [1, 1, 10, 10], "time_s": [0.0, 100.0, 900.0, 1000.0]})
    candidates = build_evidence_candidates({"cycle_features": cycle_features, "normalized_timeseries": normalized})
    retention = candidates.loc[candidates["feature_name"] == "discharge_capacity_retention"].iloc[0]
    assert retention["contract_id"].startswith("bfl:cycle_features:")
    assert retention["definition"] and retention["method"]
    assert json.loads(retention["inputs"]) == ["discharge_capacity_ah", "cycle_index"]
    assert retention["source_time_start_s"] == pytest.approx(0.0)
    assert retention["source_time_end_s"] == pytest.approx(1000.0)


def test_selector_budget() -> None:
    scored = pd.DataFrame([_candidate("large", "Large", score=0.95, estimated_token_cost=60), _candidate("small", "Small", score=0.85, estimated_token_cost=20)])
    selected = select_evidence(scored, EvidenceConfig(token_budget=50, max_selected_items=3))
    assert selected["evidence_id"].tolist() == ["small"]
