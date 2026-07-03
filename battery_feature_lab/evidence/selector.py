"""Budgeted greedy evidence selection with semantic redundancy control."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from battery_feature_lab.evidence.schema import empty_evidence_frame, normalize_evidence_frame
from battery_feature_lab.schemas import EvidenceConfig


def select_evidence(
    scored_candidates: pd.DataFrame,
    config: EvidenceConfig | None = None,
) -> pd.DataFrame:
    """Select evidence by marginal score under item, token, and redundancy constraints."""

    config = config or EvidenceConfig()
    if scored_candidates is None or scored_candidates.empty:
        return empty_evidence_frame()

    candidates = scored_candidates.reset_index(drop=True).copy()
    candidates["_stable_id"] = candidates.apply(_stable_id, axis=1)
    costs = candidates.apply(_effective_token_cost, axis=1).to_numpy(dtype=int)
    similarities = _pairwise_similarity(candidates)
    selected_positions: list[int] = []
    used_budget = 0
    source_counts: dict[str, int] = {}
    selection_details: dict[int, tuple[float, float, float]] = {}

    while len(selected_positions) < config.max_selected_items:
        best: tuple[float, float, str, int, float] | None = None

        for position, row in candidates.iterrows():
            if position in selected_positions:
                continue

            base_score = _finite_float(row.get("score"), default=0.0)
            if base_score < config.min_score:
                continue
            if config.token_budget and used_budget + costs[position] > config.token_budget:
                continue

            source = str(row.get("source_table") or "")
            if (
                config.max_items_per_source is not None
                and source_counts.get(source, 0) >= config.max_items_per_source
            ):
                continue

            max_similarity = (
                float(np.max(similarities[position, selected_positions]))
                if selected_positions
                else 0.0
            )
            if max_similarity >= config.redundancy_threshold:
                continue

            redundancy_penalty = config.selection_redundancy_weight * max_similarity
            marginal_score = base_score - redundancy_penalty
            rank_key = (
                marginal_score,
                base_score,
                str(row["_stable_id"]),
                position,
                max_similarity,
            )
            if best is None or rank_key[:2] > best[:2] or (
                rank_key[:2] == best[:2] and rank_key[2] < best[2]
            ):
                best = rank_key

        if best is None:
            break

        marginal_score, base_score, _, position, max_similarity = best
        selected_positions.append(position)
        used_budget += costs[position]
        source = str(candidates.iloc[position].get("source_table") or "")
        source_counts[source] = source_counts.get(source, 0) + 1
        selection_details[position] = (
            config.selection_redundancy_weight * max_similarity,
            marginal_score,
            max_similarity,
        )

    if not selected_positions:
        return empty_evidence_frame()

    selected = candidates.iloc[selected_positions].drop(columns=["_stable_id"]).copy()
    selected["estimated_token_cost"] = costs[selected_positions]
    selected["selected"] = True
    selected["selection_rank"] = range(1, len(selected) + 1)
    selected["redundancy_penalty"] = [selection_details[pos][0] for pos in selected_positions]
    selected["selection_score"] = [selection_details[pos][1] for pos in selected_positions]
    selected["selected_similarity"] = [selection_details[pos][2] for pos in selected_positions]
    return normalize_evidence_frame(selected)


def _pairwise_similarity(candidates: pd.DataFrame) -> np.ndarray:
    texts = candidates.apply(_selection_text, axis=1).tolist()
    if not texts or not any(text.strip() for text in texts):
        return np.eye(len(texts), dtype=float)
    try:
        matrix = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            norm="l2",
        ).fit_transform(texts)
        return cosine_similarity(matrix)
    except ValueError:
        return np.eye(len(texts), dtype=float)


def _selection_text(row: pd.Series) -> str:
    return " ".join(
        str(row.get(field))
        for field in ("text", "interpretation_hint", "support_role")
        if not _missing(row.get(field))
    )


def _effective_token_cost(row: pd.Series) -> int:
    for field in ("estimated_token_cost", "token_cost"):
        value = _finite_float(row.get(field), default=0.0)
        if value > 0:
            return int(np.ceil(value))
    return 0


def _stable_id(row: pd.Series) -> str:
    return str(row.get("evidence_id") or row.get("redundancy_key") or row.name)


def _finite_float(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _missing(value: object) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
