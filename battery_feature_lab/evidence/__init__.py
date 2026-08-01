"""Evidence-layer utilities for question-aware context selection."""

from battery_feature_lab.evidence.candidates import build_evidence_candidates
from battery_feature_lab.evidence.scorer import score_evidence_candidates
from battery_feature_lab.evidence.selector import select_evidence

__all__ = ["build_evidence_candidates", "score_evidence_candidates", "select_evidence"]
