"""EIS compact index: shared conditions and interpretation limits stated once."""

from __future__ import annotations

from typing import Any

EIS_SUMMARY_SCHEMA_VERSION = "bfl.eis-summary/0.2"
EMPTY_MEMBER_FIELDS = (
    "member_record_count",
    "member_record_ids",
    "member_summaries",
    "member_summaries_status",
)
RETRIEVAL_INSTRUCTIONS = (
    "Read this file first. Conditions identical for every spectrum are listed once under "
    "conditions.shared, and conditions reported for no spectrum under conditions.not_reported; "
    "operation records list only the conditions that differ between spectra. Interpretation "
    "limits are listed once and referenced by identifier. Resolve a record_id in "
    "analysis_evidence.json for complete condition blocks, source intervals, derivation details, "
    "parameters, references, or full series."
)


def compact_eis_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove repetition from the generic compact index without dropping information.

    Every value removed from a compact record remains available from the run-level
    ``conditions`` and ``interpretation_limits`` blocks or from the evidence record
    with the same ``record_id``.
    """

    operations = summary["dimensions"]["operation"]
    records = [record for items in summary["dimensions"].values() for record in items]
    shared: dict[str, Any] = {}
    not_reported: list[str] = []
    if operations:
        for name, block in operations[0]["attributes"]["conditions"].items():
            if all(op["attributes"]["conditions"].get(name) == block for op in operations):
                if block["status"] == "not_reported":
                    not_reported.append(name)
                else:
                    shared[name] = block
    common = set(shared) | set(not_reported)
    spectrum_ids = {
        op["attributes"]["spectrum_uid"]: op["attributes"]["spectrum_id"] for op in operations
    }
    limits: dict[str, str] = {}
    for record in records:
        attributes = record["attributes"]
        if not attributes.get("member_record_ids") and not attributes.get("member_summaries"):
            for name in EMPTY_MEMBER_FIELDS:
                attributes.pop(name, None)
        if record["record_type"] == "operation.eis_conditions":
            attributes["conditions"] = {
                name: block
                for name, block in attributes["conditions"].items()
                if name not in common
            }
        if record["record_type"] == "evolution.eis_comparison":
            attributes.pop("conditions", None)
            uids = record["record_id"].split(":", 1)[1].split(":")
            attributes["compared_spectra"] = [
                {"spectrum_uid": uid, "spectrum_id": spectrum_ids.get(uid)} for uid in uids
            ]
        record["interpretation_limits"] = [
            limits.setdefault(text, f"L{len(limits) + 1}")
            for text in record["interpretation_limits"]
        ]
    summary["schema_version"] = EIS_SUMMARY_SCHEMA_VERSION
    summary["conditions"] = {"shared": shared, "not_reported": sorted(not_reported)}
    summary["interpretation_limits"] = {key: text for text, key in limits.items()}
    summary["retrieval"]["instructions"] = RETRIEVAL_INSTRUCTIONS
    return summary
