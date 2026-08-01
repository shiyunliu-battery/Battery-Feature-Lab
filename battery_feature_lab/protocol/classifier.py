"""Recognize the test protocol of a cycle from its primitive step sequence.

Design note (open-set, not a fixed list):
    There are far more protocols in the wild than any hardcoded enum can hold
    (CC, CV, CC-CV, GITT, HPPC, DST/WLTP/UDDS and other drive cycles, RPT,
    formation, calendar rest, dQ/dV sweeps, multi-stage charge, ...). So this
    recognizer does NOT assume a closed set:

      1. The primitive step layer (rest / CC / CV / pulse / dynamic) from
         :mod:`segmenter` is universal and always describes the data.
      2. Named protocols live in an *extensible registry* (:data:`TEMPLATES`).
         Add a template to recognize a new protocol without touching the core.
      3. Anything not matched is never force-labelled. It gets
         ``protocol = "other"`` plus a structural ``signature`` (e.g.
         ``"cc_charge>cv_charge>rest"`` or ``"[pulse_discharge>rest]x12"``) and a
         coarse ``protocol_family``, so the output stays honest and useful.

The result for one cycle is a dict:
    ``{"protocol", "protocol_family", "signature", "confidence"}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from battery_feature_lab.protocol.segmenter import (
    CC_CHARGE,
    CC_DISCHARGE,
    CV_CHARGE,
    CV_DISCHARGE,
    DYNAMIC,
    PULSE_CHARGE,
    PULSE_DISCHARGE,
    REST,
)
from battery_feature_lab.schemas import ProtocolConfig

# Protocol families (coarse buckets; open enough to place unknown protocols).
FAMILY_REST = "rest"
FAMILY_CHARGE = "charge"
FAMILY_DISCHARGE = "discharge"
FAMILY_CCCV = "cccv"
FAMILY_PULSE = "pulse_characterization"
FAMILY_DYNAMIC = "dynamic"
FAMILY_MIXED = "mixed"

# Named protocols recognized by the initial registry.
REST_PROTOCOL = "rest"
CONSTANT_CURRENT = "constant_current"
CC_CV = "cc_cv"
GITT = "gitt"
HPPC = "hppc"
DST = "dst"
PULSE_TRAIN = "pulse_train"
OTHER = "other"
UNKNOWN = "unknown"

_CV_LABELS = {CV_CHARGE, CV_DISCHARGE}
_CC_LABELS = {CC_CHARGE, CC_DISCHARGE}
_PULSE_LABELS = {PULSE_CHARGE, PULSE_DISCHARGE}


def classify_protocol(
    segments: list[dict[str, Any]],
    config: ProtocolConfig,
) -> dict[str, Any]:
    """Return an open-set protocol result for one cycle's segments."""

    if not segments:
        return {
            "protocol": UNKNOWN,
            "protocol_family": FAMILY_MIXED,
            "signature": "",
            "confidence": 0.0,
            "confidence_kind": "heuristic",
            "match_reasons": ["No protocol segments were available."],
        }

    stats = _segment_stats(segments, config)
    signature = build_signature([seg["step_label"] for seg in segments])

    matches: list[tuple[ProtocolTemplate, ProtocolMatch]] = []
    for template in TEMPLATES:
        matched = template.match(stats, config)
        if matched is not None:
            matches.append((template, matched))

    if matches:
        template, matched = max(matches, key=lambda item: item[1].confidence)
        return {
            "protocol": template.name,
            "protocol_family": matched.family or template.family,
            "signature": signature,
            "confidence": round(min(0.95, matched.confidence), 4),
            "confidence_kind": "heuristic",
            "match_reasons": list(matched.reasons),
        }

    # Unmatched: stay honest — describe structure, infer a family, don't force a name.
    return {
        "protocol": OTHER,
        "protocol_family": _infer_family(stats),
        "signature": signature,
        "confidence": 0.3,
        "confidence_kind": "heuristic",
        "match_reasons": ["No named protocol template satisfied all structural constraints."],
    }


# --------------------------------------------------------------------------- #
# Structural signature
# --------------------------------------------------------------------------- #
def build_signature(labels: list[str]) -> str:
    """Collapse a label sequence into a compact structural signature.

    Consecutive duplicates are merged, and a repeating motif (period 1-3 repeated
    >= 3 times) is expressed as ``[motif]xN``.
    """

    collapsed = _collapse_repeats(labels)
    if not collapsed:
        return ""
    motif = _detect_motif(collapsed)
    if motif is not None:
        pattern, repeats, tail = motif
        base = f"[{'>'.join(pattern)}]x{repeats}"
        return base if not tail else f"{base}>{'>'.join(tail)}"
    return ">".join(collapsed)


def _collapse_repeats(labels: list[str]) -> list[str]:
    collapsed: list[str] = []
    for label in labels:
        if not collapsed or collapsed[-1] != label:
            collapsed.append(label)
    return collapsed


def _detect_motif(labels: list[str]) -> tuple[list[str], int, list[str]] | None:
    n = len(labels)
    for period in (1, 2, 3):
        if n < period * 3:
            continue
        pattern = labels[:period]
        repeats = 0
        i = 0
        while i + period <= n and labels[i : i + period] == pattern:
            repeats += 1
            i += period
        if repeats >= 3:
            return pattern, repeats, labels[i:]
    return None


# --------------------------------------------------------------------------- #
# Per-cycle statistics used by templates
# --------------------------------------------------------------------------- #
def _segment_stats(segments: list[dict[str, Any]], config: ProtocolConfig) -> dict[str, Any]:
    labels = [seg["step_label"] for seg in segments]
    non_rest = [seg for seg in segments if seg["step_label"] != REST]
    non_rest_points = sum(int(seg["n_points"]) for seg in non_rest) or 1
    dynamic_points = sum(int(seg["n_points"]) for seg in segments if seg["step_label"] == DYNAMIC)
    pulses = [seg for seg in segments if seg["step_label"] in _PULSE_LABELS]
    return {
        "labels": labels,
        "n_segments": len(segments),
        "n_non_rest": len(non_rest),
        "n_cc": sum(1 for lbl in labels if lbl in _CC_LABELS),
        "n_cv": sum(1 for lbl in labels if lbl in _CV_LABELS),
        "n_rest": sum(1 for lbl in labels if lbl == REST),
        "n_dynamic": sum(1 for lbl in labels if lbl == DYNAMIC),
        "n_pulses": len(pulses),
        "n_charge_pulses": sum(1 for seg in pulses if seg["step_label"] == PULSE_CHARGE),
        "n_discharge_pulses": sum(
            1 for seg in pulses if seg["step_label"] == PULSE_DISCHARGE
        ),
        "pulse_directions": [seg["step_label"] for seg in pulses],
        "has_charge_pulse": any(seg["step_label"] == PULSE_CHARGE for seg in pulses),
        "has_discharge_pulse": any(seg["step_label"] == PULSE_DISCHARGE for seg in pulses),
        "pulses_paired_with_rest": _pulses_paired_with_rest(segments),
        "dynamic_fraction": dynamic_points / non_rest_points,
        "has_cc_then_cv": _has_cc_then_cv(labels),
    }


def _has_cc_then_cv(labels: list[str]) -> bool:
    return any(
        (left, right)
        in {(CC_CHARGE, CV_CHARGE), (CC_DISCHARGE, CV_DISCHARGE)}
        for left, right in zip(labels, labels[1:])
    )


def _pulses_paired_with_rest(segments: list[dict[str, Any]]) -> int:
    paired = 0
    for position, segment in enumerate(segments):
        if segment["step_label"] not in _PULSE_LABELS:
            continue
        prev_rest = position > 0 and segments[position - 1]["step_label"] == REST
        next_rest = position + 1 < len(segments) and segments[position + 1]["step_label"] == REST
        if prev_rest or next_rest:
            paired += 1
    return paired


def _infer_family(stats: dict[str, Any]) -> str:
    if stats["n_non_rest"] == 0:
        return FAMILY_REST
    if stats["dynamic_fraction"] >= 0.2:
        return FAMILY_DYNAMIC
    if stats["n_pulses"] >= 1:
        return FAMILY_PULSE
    active_labels = {label for label in stats["labels"] if label != REST}
    if active_labels and active_labels <= {CC_CHARGE, CV_CHARGE}:
        return FAMILY_CHARGE
    if active_labels and active_labels <= {CC_DISCHARGE, CV_DISCHARGE}:
        return FAMILY_DISCHARGE
    return FAMILY_MIXED


# --------------------------------------------------------------------------- #
# Extensible template registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProtocolMatch:
    confidence: float
    reasons: tuple[str, ...]
    family: str | None = None


class ProtocolTemplate:
    """A named protocol rule. ``match`` returns a confidence or ``None``."""

    def __init__(
        self,
        name: str,
        family: str,
        matcher: Callable[[dict[str, Any], ProtocolConfig], ProtocolMatch | None],
    ) -> None:
        self.name = name
        self.family = family
        self._matcher = matcher

    def match(
        self, stats: dict[str, Any], config: ProtocolConfig
    ) -> ProtocolMatch | None:
        return self._matcher(stats, config)


def _match_rest(stats: dict[str, Any], config: ProtocolConfig) -> ProtocolMatch | None:
    if stats["n_non_rest"] != 0:
        return None
    return ProtocolMatch(0.9, ("All segments are zero-current rest segments.",))


def _match_hppc(stats: dict[str, Any], config: ProtocolConfig) -> ProtocolMatch | None:
    minimum = config.hppc_min_pulses_per_direction
    directions = stats["pulse_directions"]
    alternating = all(left != right for left, right in zip(directions, directions[1:]))
    if (
        stats["n_charge_pulses"] >= minimum
        and stats["n_discharge_pulses"] >= minimum
        and stats["pulses_paired_with_rest"] == stats["n_pulses"]
        and alternating
    ):
        confidence = 0.72 + 0.18 * min(1.0, stats["n_pulses"] / 8)
        return ProtocolMatch(
            confidence,
            (
                f"At least {minimum} qualified pulses occur in each direction.",
                "Pulse directions alternate and every pulse is paired with a qualified rest.",
            ),
        )
    return None


def _match_gitt(stats: dict[str, Any], config: ProtocolConfig) -> ProtocolMatch | None:
    unidirectional = not (stats["has_charge_pulse"] and stats["has_discharge_pulse"])
    if (
        stats["n_pulses"] >= config.gitt_min_pulses
        and unidirectional
        and stats["pulses_paired_with_rest"] == stats["n_pulses"]
        and stats["n_non_rest"] == stats["n_pulses"]
        and stats["n_rest"] >= stats["n_pulses"] - 1
    ):
        confidence = 0.72 + 0.18 * min(1.0, stats["n_pulses"] / 8)
        return ProtocolMatch(
            confidence,
            (
                "All active segments are qualified, unidirectional pulses.",
                "Each pulse is paired with a sufficiently long relaxation segment.",
            ),
        )
    return None


def _match_pulse_train(
    stats: dict[str, Any], config: ProtocolConfig
) -> ProtocolMatch | None:
    # Generic pulse+rest characterization that did not match GITT/HPPC exactly.
    if stats["n_pulses"] >= 2 and stats["pulses_paired_with_rest"] >= 2:
        return ProtocolMatch(
            0.5,
            ("At least two qualified pulse/rest pairs are present.",),
        )
    return None


def _match_cc_cv(stats: dict[str, Any], config: ProtocolConfig) -> ProtocolMatch | None:
    if not stats["has_cc_then_cv"] or stats["n_dynamic"] or stats["n_pulses"]:
        return None
    return ProtocolMatch(
        0.85,
        ("An adjacent same-direction CC step and voltage-validated CV tail are present.",),
    )


def _match_constant_current(
    stats: dict[str, Any], config: ProtocolConfig
) -> ProtocolMatch | None:
    if stats["n_cc"] == 1 and stats["n_pulses"] == 0 and stats["n_cv"] == 0:
        label = next(label for label in stats["labels"] if label in _CC_LABELS)
        family = FAMILY_CHARGE if label == CC_CHARGE else FAMILY_DISCHARGE
        return ProtocolMatch(
            0.75,
            (f"Exactly one constant-current {family} segment is present.",),
            family=family,
        )
    return None


# Order matters: more specific templates first. Extend this list to teach the
# recognizer new protocols (e.g. WLTP, UDDS, RPT, formation, multi-stage charge).
TEMPLATES: list[ProtocolTemplate] = [
    ProtocolTemplate(REST_PROTOCOL, FAMILY_REST, _match_rest),
    ProtocolTemplate(HPPC, FAMILY_PULSE, _match_hppc),
    ProtocolTemplate(GITT, FAMILY_PULSE, _match_gitt),
    ProtocolTemplate(PULSE_TRAIN, FAMILY_PULSE, _match_pulse_train),
    ProtocolTemplate(CC_CV, FAMILY_CCCV, _match_cc_cv),
    ProtocolTemplate(CONSTANT_CURRENT, FAMILY_MIXED, _match_constant_current),
]
