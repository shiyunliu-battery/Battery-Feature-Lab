"""Question-aware evidence scoring.

This module scores evidence candidates using algorithmic constraints rather than
fixed domain priors. The scorer is designed to work without a trained model and
without external LLM calls.

Main ideas:
- Relevance is estimated by TF-IDF cosine similarity between the user question
  and each evidence text.
- Reliability is derived only from explicit numeric confidence and uncertainty fields.
- Protocol support is derived from protocol metadata and optional protocol
  confidence columns.
- Redundancy is handled as a marginal penalty during greedy selection.
- Token cost is treated as a penalty, but selection should still enforce the
  final token budget.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from battery_feature_lab.evidence.schema import empty_evidence_frame, normalize_evidence_frame
from battery_feature_lab.schemas import EvidenceConfig


_EPS = 1e-12


def score_evidence_candidates(
    candidates: pd.DataFrame,
    config: EvidenceConfig | None = None,
) -> pd.DataFrame:
    """Score evidence candidates for a user question."""

    config = config or EvidenceConfig()

    if candidates is None or candidates.empty:
        return empty_evidence_frame()

    scored = candidates.copy()
    question = _clean_text(config.question)

    similarity = _TextSimilarity()
    question_counts = similarity.counts(question) if question else None

    scored["_evidence_text"] = scored.apply(_build_evidence_text, axis=1)
    scored["question_relevance"] = _semantic_relevance(
        similarity, question_counts, scored["_evidence_text"]
    )
    scored["lexical_overlap"] = _lexical_overlap(question, scored["_evidence_text"])
    scored["reliability_score"] = scored.apply(_reliability_score, axis=1)
    scored["metadata_completeness"] = scored.apply(_evidence_metadata_completeness, axis=1)
    scored["protocol_available"] = scored.apply(_protocol_available, axis=1)
    scored["protocol_consistency"] = scored.apply(
        lambda row: _protocol_consistency(row, question, similarity, question_counts),
        axis=1,
    )
    scored["informativeness"] = scored.apply(_informativeness_score, axis=1)
    scored["source_confidence"] = scored.apply(_source_confidence, axis=1)
    scored["estimated_token_cost"] = scored.apply(_estimated_token_cost, axis=1)
    scored["token_penalty"] = _token_penalty(scored["estimated_token_cost"], config)
    # Redundancy is conditional on what has already been selected. It belongs in the
    # greedy selector rather than the candidate's intrinsic score.
    scored["redundancy_penalty"] = 0.0

    component_columns = [
        "question_relevance",
        "lexical_overlap",
        "reliability_score",
        "protocol_consistency",
        "informativeness",
        "source_confidence",
    ]

    scored["evidence_quality"] = scored.apply(
        lambda row: _adaptive_component_score(row, component_columns, config),
        axis=1,
    )

    scored["score"] = (
        scored["evidence_quality"]
        - scored["token_penalty"]
    ).clip(lower=0.0, upper=1.0)

    scored["score"] = scored["score"].round(6)
    scored["selected"] = False
    scored["selection_rank"] = None

    scored = scored.drop(columns=["_evidence_text"], errors="ignore")
    return normalize_evidence_frame(scored)


class _TextSimilarity:
    """Closed-form TF-IDF cosine for question/evidence text pairs.

    Reproduces exactly what a ``TfidfVectorizer`` fitted on the two texts of a
    single pair would return, so relevance stays candidate-independent (adding a
    candidate never changes another's score). It shares one analyzer across all
    candidates instead of fitting a fresh vectorizer per row, which turns the
    previous O(n) sklearn fits into a single analyzer build plus cheap token
    counting.
    """

    def __init__(self) -> None:
        self._analyzer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
        ).build_analyzer()
        # smooth_idf over two documents: a term shared by both (df=2) has
        # idf ln(3/3)+1 = 1.0; a term in only one document (df=1) has
        # idf ln(3/2)+1. Only shared terms contribute to the dot product.
        self._idf_single = math.log(3.0 / 2.0) + 1.0

    def counts(self, text: Any) -> Counter:
        return Counter(self._analyzer(_clean_text(text)))

    def similarity_from_counts(self, left_counts: Counter, right: Any) -> float:
        return self._similarity(left_counts, self.counts(right))

    def _similarity(self, left: Counter, right: Counter) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(count * right[term] for term, count in left.items() if term in right)
        if numerator == 0.0:
            return 0.0
        left_norm = math.sqrt(
            sum(
                (count * (1.0 if term in right else self._idf_single)) ** 2
                for term, count in left.items()
            )
        )
        right_norm = math.sqrt(
            sum(
                (count * (1.0 if term in left else self._idf_single)) ** 2
                for term, count in right.items()
            )
        )
        denominator = left_norm * right_norm
        return numerator / denominator if denominator else 0.0


def _semantic_relevance(
    similarity: _TextSimilarity,
    question_counts: Counter | None,
    evidence_texts: pd.Series,
) -> np.ndarray:
    """Estimate question-evidence relevance using TF-IDF cosine similarity."""

    texts = evidence_texts.fillna("").astype(str).tolist()

    if not texts:
        return np.array([], dtype=float)

    if question_counts is None:
        return np.full(len(texts), np.nan, dtype=float)

    return np.asarray(
        [similarity.similarity_from_counts(question_counts, text) for text in texts],
        dtype=float,
    )


def _lexical_overlap(question: str, evidence_texts: pd.Series) -> np.ndarray:
    """Compute token overlap between the question and each evidence text."""

    if not question:
        return np.full(len(evidence_texts), np.nan, dtype=float)

    query_terms = _content_tokens(question)
    if not query_terms:
        return np.full(len(evidence_texts), np.nan, dtype=float)

    scores: list[float] = []
    for text in evidence_texts.fillna("").astype(str):
        evidence_terms = _content_tokens(text)
        if not evidence_terms:
            scores.append(0.0)
            continue
        scores.append(len(query_terms & evidence_terms) / max(len(query_terms), 1))

    return np.asarray(scores, dtype=float)


def _reliability_score(row: pd.Series) -> float:
    """Derive reliability only from explicit numeric quality signals."""

    numeric_confidences = _numeric_values(
        row,
        [
            "reliability_score",
            "confidence",
            "support_confidence",
            "feature_confidence",
            "quality_score",
            "fit_r2",
            "r2",
        ],
    )
    uncertainty_penalty = _uncertainty_penalty(row)

    score = (
        float(np.nanmean([_bounded(v) for v in numeric_confidences]))
        if numeric_confidences
        else 0.5
    )
    return _bounded(score - uncertainty_penalty)


def _evidence_metadata_completeness(row: pd.Series) -> float:
    """Report metadata completeness separately from scientific reliability."""

    return _metadata_completeness_score(
        row,
        ["source_table", "support_role", "feature_name", "evidence_type", "text"],
    )


def _protocol_available(row: pd.Series) -> bool:
    return any(
        _clean_text(row.get(field)) not in {"", "unknown", "none", "nan"}
        for field in ("protocol", "protocol_family")
    )


def _protocol_consistency(
    row: pd.Series,
    question: str,
    similarity: _TextSimilarity,
    question_counts: Counter | None,
) -> float:
    """Estimate whether protocol metadata supports the evidence context."""

    protocol = _clean_text(row.get("protocol"))
    protocol_family = _clean_text(row.get("protocol_family"))
    protocol_text = " ".join(
        value
        for value in [protocol, protocol_family]
        if value and value not in {"unknown", "none", "nan"}
    )

    confidence_values = _numeric_values(
        row,
        ["protocol_confidence", "segment_confidence", "classification_confidence"],
    )

    if not protocol_text:
        return float("nan")

    if confidence_values:
        confidence_score = float(np.nanmean([_bounded(v) for v in confidence_values]))
    else:
        confidence_score = 0.5

    if question_counts is None:
        return _bounded(confidence_score)

    semantic_score = similarity.similarity_from_counts(question_counts, protocol_text)

    return _bounded(0.65 * confidence_score + 0.35 * semantic_score)


def _informativeness_score(row: pd.Series) -> float:
    """Estimate whether the evidence contains enough information to be useful."""

    text = _build_evidence_text(row)
    token_count = len(_content_tokens(text))

    fields = [
        "source_table",
        "support_role",
        "feature_name",
        "evidence_type",
        "protocol",
        "cycle_index",
        "segment_id",
        "value",
        "unit",
        "interpretation_hint",
    ]

    completeness = _metadata_completeness_score(row, fields)
    numeric_signal = 1.0 if _row_has_numeric_signal(row) else 0.0
    text_signal = _bounded(math.log1p(token_count) / math.log1p(80))

    return _bounded(0.45 * completeness + 0.35 * text_signal + 0.20 * numeric_signal)


def _source_confidence(row: pd.Series) -> float:
    """Use source confidence only when the producer supplies a numeric value."""

    values = _numeric_values(row, ["source_confidence", "source_quality"])
    return _bounded(float(np.nanmean(values))) if values else float("nan")


def _token_penalty(token_cost: pd.Series, config: EvidenceConfig) -> pd.Series:
    """Convert token cost into a bounded penalty."""

    values = pd.to_numeric(token_cost, errors="coerce").fillna(0.0).clip(lower=0.0)

    cap = float(config.token_penalty_cap)

    if values.empty or float(values.max()) <= 0:
        return pd.Series(np.zeros(len(values)), index=values.index)

    if config.token_budget <= 0 or config.max_selected_items <= 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    scale = max(config.token_budget / config.max_selected_items, 1.0)

    penalties = cap * np.sqrt(values / scale)
    return penalties.clip(lower=0.0, upper=cap)


def _adaptive_component_score(
    row: pd.Series,
    component_columns: list[str],
    config: EvidenceConfig,
) -> float:
    """Combine score components using optional config weights or adaptive averaging."""

    weights = config.scoring_weights

    values: list[float] = []
    row_weights: list[float] = []

    for column in component_columns:
        value = row.get(column)

        if value is None or pd.isna(value):
            continue

        values.append(_bounded(float(value)))

        if isinstance(weights, dict) and column in weights:
            row_weights.append(max(float(weights[column]), 0.0))
        else:
            row_weights.append(1.0)

    if not values:
        return 0.0

    weight_array = np.asarray(row_weights, dtype=float)
    value_array = np.asarray(values, dtype=float)

    if float(weight_array.sum()) <= 0:
        return float(value_array.mean())

    arithmetic = float(np.average(value_array, weights=weight_array))
    geometric = float(np.exp(np.average(np.log(value_array + _EPS), weights=weight_array)))

    blend_total = config.arithmetic_score_weight + config.geometric_score_weight
    return _bounded(
        (
            config.arithmetic_score_weight * arithmetic
            + config.geometric_score_weight * geometric
        )
        / blend_total
    )


def _build_evidence_text(row: pd.Series) -> str:
    """Build a compact text representation of an evidence candidate."""

    fields = [
        "text",
        "interpretation_hint",
        "support_role",
        "feature_name",
        "evidence_type",
        "protocol",
        "protocol_family",
        "cycle_index",
        "segment_id",
        "step_index",
        "value",
        "unit",
        "trend",
        "direction",
        "summary",
        "description",
        "reason",
        "warning",
        "reliability",
    ]

    parts: list[str] = []

    for field in fields:
        if field not in row.index:
            continue

        value = row.get(field)

        if _is_missing(value):
            continue

        parts.append(str(value))

    return " | ".join(parts)


def _estimated_token_cost(row: pd.Series) -> int:
    """Estimate token cost when an explicit token_cost column is unavailable."""

    explicit = row.get("token_cost")

    if not _is_missing(explicit):
        try:
            value = int(float(explicit))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass

    text = _build_evidence_text(row)

    if not text:
        return 0

    return max(1, math.ceil(len(text) / 4))


def _metadata_completeness_score(row: pd.Series, fields: Iterable[str]) -> float:
    """Estimate metadata completeness for a row."""

    fields = list(fields)

    if not fields:
        return 0.5

    present = 0

    for field in fields:
        if field in row.index and not _is_unavailable_metadata(row.get(field)):
            present += 1

    return present / len(fields)


def _uncertainty_penalty(row: pd.Series) -> float:
    """Estimate uncertainty penalty from optional uncertainty-like fields."""

    uncertainty_values = _numeric_values(
        row,
        [
            "uncertainty",
            "std",
            "standard_error",
            "measurement_error",
            "missing_fraction",
            "nan_fraction",
        ],
    )

    if not uncertainty_values:
        return 0.0

    bounded_values = [_bounded(v) for v in uncertainty_values]
    return 0.25 * float(np.nanmean(bounded_values))


def _numeric_values(row: pd.Series, fields: Iterable[str]) -> list[float]:
    """Extract finite numeric values from row fields."""

    values: list[float] = []

    for field in fields:
        if field not in row.index:
            continue

        value = row.get(field)

        if _is_missing(value):
            continue

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(numeric):
            values.append(numeric)

    return values


def _row_has_numeric_signal(row: pd.Series) -> bool:
    """Return True when the row contains at least one finite numeric signal."""

    excluded = {
        # Scoring outputs produced by this module.
        "score",
        "question_relevance",
        "lexical_overlap",
        "reliability_score",
        "protocol_consistency",
        "informativeness",
        "source_confidence",
        "token_penalty",
        "redundancy_penalty",
        "selection_rank",
        "selected",
        # Cost and confidence/quality metadata. These describe the evidence,
        # they are not a battery measurement contained in the evidence, so they
        # must not be read as an intrinsic numeric signal.
        "token_cost",
        "estimated_token_cost",
        "confidence",
        "support_confidence",
        "feature_confidence",
        "quality_score",
        "protocol_confidence",
        "segment_confidence",
        "classification_confidence",
        "source_quality",
        "fit_r2",
        "r2",
        "uncertainty",
        "std",
        "standard_error",
        "measurement_error",
        "missing_fraction",
        "nan_fraction",
    }

    for column, value in row.items():
        if column in excluded:
            continue

        if _is_missing(value):
            continue

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(numeric):
            return True

    return False


def _content_tokens(text: str) -> set[str]:
    """Tokenize text into lightweight content tokens."""

    text = _clean_text(text)
    tokens = re.findall(r"[a-zA-Z0-9_\-]+", text)

    return {
        token
        for token in tokens
        if len(token) >= 2 and not token.isnumeric()
    }


def _clean_text(value: Any) -> str:
    """Convert a value to normalized lowercase text."""

    if _is_missing(value):
        return ""

    return str(value).strip().lower()


def _is_missing(value: Any) -> bool:
    """Return True for None, NaN, empty strings and empty collections."""

    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    if isinstance(value, str) and not value.strip():
        return True

    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return False

    return False


def _is_unavailable_metadata(value: Any) -> bool:
    if _is_missing(value):
        return True
    return _clean_text(value) in {"unknown", "none", "nan", "n/a", "na"}


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Bound a numeric value to a closed interval."""

    if not math.isfinite(value):
        return lower

    return min(max(value, lower), upper)
