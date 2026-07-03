"""Shared evidence schema helpers."""

from __future__ import annotations

import pandas as pd

EVIDENCE_COLUMNS = [
    "cell_id",
    "evidence_id",
    "evidence_type",
    "source_table",
    "source_row_index",
    "cycle_index",
    "cycle_start",
    "cycle_end",
    "feature_name",
    "value",
    "unit",
    "protocol",
    "protocol_family",
    "protocol_confidence",
    "support_role",
    "reliability",
    "interpretation_hint",
    "text",
    "token_cost",
    "estimated_token_cost",
    "redundancy_key",
    "question_relevance",
    "lexical_overlap",
    "protocol_consistency",
    "protocol_available",
    "reliability_score",
    "metadata_completeness",
    "informativeness",
    "source_confidence",
    "token_penalty",
    "redundancy_penalty",
    "evidence_quality",
    "score",
    "selection_score",
    "selected_similarity",
    "selection_rank",
    "selected",
]


def empty_evidence_frame() -> pd.DataFrame:
    """Return an empty evidence frame with stable columns."""

    return pd.DataFrame(columns=EVIDENCE_COLUMNS)


def normalize_evidence_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure stable core columns without discarding source-specific metadata."""

    normalized = frame.copy()
    for column in EVIDENCE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    extra_columns = sorted(
        column for column in normalized.columns if column not in EVIDENCE_COLUMNS
    )
    return normalized[[*EVIDENCE_COLUMNS, *extra_columns]]
