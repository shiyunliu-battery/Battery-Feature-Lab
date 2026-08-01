"""Protocol-aware segmentation and recognition for battery time series.

Turns a normalized time series into labelled protocol segments (CC-CV, GITT,
HPPC, DST, rest, ...) and provides an interactive region describer. Rule-based
and dependency-light (numpy/pandas/scipy only); the approach is adapted from
cellpy's step-table categorisation and ampworks' pulse descriptors.
"""

from battery_feature_lab.protocol.classifier import classify_protocol
from battery_feature_lab.protocol.detector import (
    annotate_normalized,
    build_protocol_records,
    describe_region,
    detect_protocol_segments,
)
from battery_feature_lab.protocol.jsonl_writer import write_protocol_jsonl
from battery_feature_lab.protocol.segmenter import segment_cycle

__all__ = [
    "annotate_normalized",
    "build_protocol_records",
    "classify_protocol",
    "describe_region",
    "detect_protocol_segments",
    "segment_cycle",
    "write_protocol_jsonl",
]
