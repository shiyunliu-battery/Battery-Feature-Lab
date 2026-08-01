from __future__ import annotations

import pandas as pd
import pytest

from battery_feature_lab.evidence.candidates import build_evidence_candidates
from battery_feature_lab.evidence.scorer import score_evidence_candidates
from battery_feature_lab.evidence.selector import select_evidence
from battery_feature_lab.schemas import EvidenceConfig


def _candidate(evidence_id: str, text: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "cell_id": "cell-1",
        "evidence_id": evidence_id,
        "evidence_type": "derived_feature",
        "source_table": "features",
        "feature_name": evidence_id,
        "support_role": "measurement",
        "text": text,
        "interpretation_hint": "Observed feature; no mechanism is inferred.",
        "protocol": "unknown",
        "token_cost": 20,
        "redundancy_key": evidence_id,
    }
    record.update(overrides)
    return record


def test_scoring_is_explainable_and_candidate_addition_stable() -> None:
    question = "Why did discharge capacity fade?"
    target = _candidate("capacity", "Discharge capacity faded across cycles.")
    first = score_evidence_candidates(
        pd.DataFrame([target]),
        EvidenceConfig(question=question),
    )
    expanded = score_evidence_candidates(
        pd.DataFrame([target, _candidate("temperature", "Mean temperature was stable.")]),
        EvidenceConfig(question=question),
    )

    target_expanded = expanded.loc[expanded["evidence_id"] == "capacity"].iloc[0]
    assert target_expanded["question_relevance"] == pytest.approx(
        first.iloc[0]["question_relevance"]
    )
    for column in (
        "lexical_overlap",
        "metadata_completeness",
        "informativeness",
        "estimated_token_cost",
        "token_penalty",
        "evidence_quality",
    ):
        assert column in expanded.columns
        assert pd.notna(target_expanded[column])


def test_unknown_protocol_is_not_treated_as_support() -> None:
    scored = score_evidence_candidates(
        pd.DataFrame([_candidate("capacity", "Capacity changed.")]),
        EvidenceConfig(question="Did the charge protocol change?"),
    )

    assert not bool(scored.iloc[0]["protocol_available"])
    assert pd.isna(scored.iloc[0]["protocol_consistency"])
    assert scored.iloc[0]["reliability_score"] == pytest.approx(0.5)


def test_selector_uses_estimated_cost_and_semantic_redundancy() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                "too-large", "Capacity faded sharply.", score=0.95, estimated_token_cost=60
            ),
            _candidate(
                "selected", "Capacity retention declined.", score=0.85, estimated_token_cost=20
            ),
            _candidate(
                "duplicate", "Capacity retention declined.", score=0.80, estimated_token_cost=20
            ),
        ]
    )
    selected = select_evidence(
        scored,
        EvidenceConfig(token_budget=50, max_selected_items=3, redundancy_threshold=0.95),
    )

    assert selected["evidence_id"].tolist() == ["selected"]
    assert selected["estimated_token_cost"].sum() <= 50
    assert selected.iloc[0]["selection_score"] == pytest.approx(0.85)


def test_new_numeric_feature_columns_generate_candidates_without_registration() -> None:
    table = pd.DataFrame(
        {
            "cell_id": ["cell-1", "cell-1"],
            "cycle_index": [1, 10],
            "new_model_output": [2.0, 3.5],
        }
    )
    candidates = build_evidence_candidates({"future_feature_table": table})
    generated = candidates.loc[candidates["feature_name"] == "new_model_output_change"]

    assert len(generated) == 1
    assert generated.iloc[0]["source_table"] == "future_feature_table"
    assert generated.iloc[0]["value"] == pytest.approx(1.5)


def test_protocol_metadata_is_joined_to_specialized_evidence() -> None:
    cycle_features = pd.DataFrame(
        {
            "cell_id": ["cell-1", "cell-1"],
            "cycle_index": [1, 10],
            "discharge_capacity_ah": [2.0, 1.8],
            "coulombic_efficiency": [0.99, 0.98],
        }
    )
    protocol_segments = pd.DataFrame(
        {
            "cell_id": ["cell-1", "cell-1"],
            "cycle_index": [1, 10],
            "segment_index": [0, 0],
            "protocol": ["constant_current", "constant_current"],
            "protocol_family": ["discharge", "discharge"],
            "protocol_confidence": [0.75, 0.75],
        }
    )
    candidates = build_evidence_candidates(
        {
            "cycle_features": cycle_features,
            "protocol_segments": protocol_segments,
        }
    )

    assert set(candidates["protocol"]) == {"constant_current"}
    assert set(candidates["protocol_family"]) == {"discharge"}
    assert set(candidates["protocol_confidence"]) == {0.75}
