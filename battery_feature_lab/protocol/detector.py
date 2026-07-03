"""Protocol-aware segmentation orchestrator.

Turns a normalized time series into:
  * ``protocol_segments`` — one row per primitive step with its descriptors, the
    segment ``step_label``, and the ``protocol`` recognized for its cycle.
  * an annotated copy of the normalized frame carrying ``protocol`` and
    ``segment_id`` per sample (so downstream evidence can stop reporting
    ``protocol = unknown``).

It also exposes :func:`describe_region`, which answers the interactive question
"I selected this window — which test is it, which segment, and how does the
voltage behave?" for an arbitrary ``[start_time, end_time]`` slice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from battery_feature_lab.protocol.classifier import UNKNOWN, classify_protocol
from battery_feature_lab.protocol.segmenter import segment_cycle
from battery_feature_lab.schemas import ProtocolConfig

_SEGMENT_COLUMNS = [
    "cell_id",
    "cycle_index",
    "segment_index",
    "segment_id",
    "protocol",
    "protocol_family",
    "protocol_signature",
    "protocol_confidence",
    "protocol_confidence_kind",
    "protocol_match_reasons",
    "step_label",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "n_points",
    "start_voltage_v",
    "end_voltage_v",
    "delta_voltage_v",
    "voltage_range_v",
    "voltage_slope_v_per_s",
    "mean_current_a",
    "max_abs_current_a",
    "current_sign",
    "c_rate",
    "start_row_index",
    "end_row_index",
    "source_row_indices",
]


def detect_protocol_segments(
    normalized: pd.DataFrame,
    config: ProtocolConfig | None = None,
    nominal_capacity_ah: float | None = None,
) -> pd.DataFrame:
    """Build the ``protocol_segments`` table from a normalized time series."""

    config = config or ProtocolConfig()
    if normalized is None or normalized.empty:
        return pd.DataFrame(columns=_SEGMENT_COLUMNS)

    frame = normalized.copy()
    if "cell_id" not in frame.columns:
        frame["cell_id"] = "cell"
    if "cycle_index" not in frame.columns:
        frame["cycle_index"] = 1

    rows: list[dict[str, Any]] = []
    for (cell_id, cycle_index), cycle_frame in frame.groupby(["cell_id", "cycle_index"], sort=True):
        segments = segment_cycle(cycle_frame, config, nominal_capacity_ah)
        if not segments:
            continue
        result = classify_protocol(segments, config)
        for segment in segments:
            segment_id = f"{cell_id}_c{int(cycle_index)}_s{segment['segment_index']}"
            rows.append(
                {
                    "cell_id": str(cell_id),
                    "cycle_index": int(cycle_index),
                    "segment_id": segment_id,
                    "protocol": result["protocol"],
                    "protocol_family": result["protocol_family"],
                    "protocol_signature": result["signature"],
                    "protocol_confidence": result["confidence"],
                    "protocol_confidence_kind": result.get(
                        "confidence_kind", "heuristic"
                    ),
                    "protocol_match_reasons": result.get("match_reasons", []),
                    **segment,
                }
            )

    if not rows:
        return pd.DataFrame(columns=_SEGMENT_COLUMNS)
    table = pd.DataFrame(rows)
    for column in _SEGMENT_COLUMNS:
        if column not in table.columns:
            table[column] = None
    return table[_SEGMENT_COLUMNS]


def annotate_normalized(normalized: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``normalized`` with per-sample ``protocol``/``segment_id``.

    The mapping is by row-index range of each segment, which the segmenter
    records in ``start_row_index``/``end_row_index``.
    """

    annotated = normalized.copy()
    annotated["protocol"] = "unknown"
    annotated["segment_id"] = None
    if segments is None or segments.empty:
        return annotated

    for _, segment in segments.iterrows():
        start, end = segment["start_row_index"], segment["end_row_index"]
        if start is None or end is None or start < 0:
            continue
        source_indices = segment.get("source_row_indices")
        if isinstance(source_indices, (list, tuple, np.ndarray, pd.Series)):
            index_mask = annotated.index.isin(list(source_indices))
        else:
            index_values = annotated.index.to_numpy()
            index_mask = (index_values >= start) & (index_values <= end)

        # Row indices are not globally scoped in every input. Always constrain
        # the mapping to the cycle that produced the segment.
        group_mask = np.ones(len(annotated), dtype=bool)
        if "cell_id" in annotated.columns:
            group_mask &= annotated["cell_id"].astype(str).eq(str(segment["cell_id"])).to_numpy()
        if "cycle_index" in annotated.columns:
            group_mask &= annotated["cycle_index"].eq(segment["cycle_index"]).to_numpy()
        mask = index_mask & group_mask
        annotated.loc[mask, "protocol"] = segment["protocol"]
        annotated.loc[mask, "segment_id"] = segment["segment_id"]
    return annotated


def describe_region(
    normalized: pd.DataFrame,
    start_time_s: float,
    end_time_s: float,
    config: ProtocolConfig | None = None,
    nominal_capacity_ah: float | None = None,
    cell_id: str | None = None,
) -> dict[str, Any]:
    """Describe a user-selected time window: protocol, segments, voltage behaviour.

    This runs the same segmentation on the selected slice and reports the
    recognized protocol plus a compact description of how voltage/current behave
    in each primitive step inside the window.
    """

    config = config or ProtocolConfig()
    frame = normalized.copy()
    if cell_id is not None and "cell_id" in frame.columns:
        frame = frame[frame["cell_id"] == cell_id]
    time = pd.to_numeric(frame["time_s"], errors="coerce")
    window = frame[(time >= start_time_s) & (time <= end_time_s)]
    if window.empty:
        return {
            "protocol": UNKNOWN,
            "protocol_family": "mixed",
            "signature": "",
            "protocol_confidence": 0.0,
            "n_segments": 0,
            "segments": [],
            "note": "No samples fall inside the selected window.",
        }

    segments = segment_cycle(window, config, nominal_capacity_ah)
    result = classify_protocol(segments, config)
    return {
        "protocol": result["protocol"],
        "protocol_family": result["protocol_family"],
        "signature": result["signature"],
        "protocol_confidence": result["confidence"],
        "start_time_s": float(start_time_s),
        "end_time_s": float(end_time_s),
        "n_segments": len(segments),
        "segments": [_describe_segment(seg) for seg in segments],
    }


def build_protocol_records(segments: pd.DataFrame) -> list[dict[str, Any]]:
    """Build detailed per-cycle protocol records for LLM/RAG consumption.

    One record per (cell, cycle): the recognized protocol, family, confidence and
    structural signature, plus each primitive segment with a compact description
    of how voltage and current behave. This is the LLM-facing JSON view of the
    ``protocol_segments`` table.
    """

    if segments is None or segments.empty:
        return []

    records: list[dict[str, Any]] = []
    grouped = segments.groupby(["cell_id", "cycle_index"], sort=True)
    for (cell_id, cycle_index), group in grouped:
        ordered = group.sort_values("segment_index")
        first = ordered.iloc[0]
        segment_views = []
        for _, row in ordered.iterrows():
            view = _describe_segment(row.to_dict())
            view["segment_index"] = int(row["segment_index"])
            view["segment_id"] = row["segment_id"]
            segment_views.append(view)
        records.append(
            {
                "cell_id": str(cell_id),
                "cycle_index": int(cycle_index),
                "protocol": first["protocol"],
                "protocol_family": first["protocol_family"],
                "protocol_confidence": _round(first["protocol_confidence"]),
                "protocol_confidence_kind": first.get(
                    "protocol_confidence_kind", "heuristic"
                ),
                "match_reasons": first.get("protocol_match_reasons", []),
                "signature": first["protocol_signature"],
                "n_segments": int(len(ordered)),
                "text": (
                    f"Cycle {int(cycle_index)} of cell {cell_id} is recognized as "
                    f"protocol '{first['protocol']}' (family '{first['protocol_family']}', "
                    f"confidence {_round(first['protocol_confidence'])}); "
                    f"structure {first['protocol_signature']}."
                ),
                "interpretation_hint": (
                    "Protocol labels are rule-based over the primitive step sequence. "
                    "A label of 'other' means the structure did not match a known template; "
                    "use the signature and per-segment behaviour rather than assuming a name."
                ),
                "segments": segment_views,
            }
        )
    return records


def _describe_segment(segment: dict[str, Any]) -> dict[str, Any]:
    label = segment["step_label"]
    dv = segment["delta_voltage_v"]
    slope = segment["voltage_slope_v_per_s"]
    direction = "rising" if dv > 0 else "falling" if dv < 0 else "flat"
    return {
        "step_label": label,
        "duration_s": round(segment["duration_s"], 3),
        "voltage": {
            "start_v": _round(segment["start_voltage_v"]),
            "end_v": _round(segment["end_voltage_v"]),
            "delta_v": _round(dv),
            "direction": direction,
            "slope_v_per_s": _round(slope, 8),
        },
        "current": {
            "mean_a": _round(segment["mean_current_a"]),
            "max_abs_a": _round(segment["max_abs_current_a"]),
            "c_rate": _round(segment["c_rate"]),
        },
        "text": (
            f"{label}: voltage {direction} from {_round(segment['start_voltage_v'])} V to "
            f"{_round(segment['end_voltage_v'])} V over {round(segment['duration_s'], 1)} s "
            f"at mean current {_round(segment['mean_current_a'])} A."
        ),
    }


def _round(value: Any, digits: int = 5) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)
