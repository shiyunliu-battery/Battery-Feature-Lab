"""Compatibility import for the evidence-only LLM context writer.

The pre-0.4 implementation built ``llm_context.jsonl`` directly from feature tables.
That path is intentionally removed: AI-visible context must pass through Feature
Contracts and evidence records first.

Importing ``build_llm_context_records`` or ``write_llm_jsonl`` from this historical
module remains supported, but both names now resolve to the evidence-only writer.
"""

from battery_feature_lab.export.evidence_context_writer import (
    build_llm_context_records,
    write_llm_jsonl,
)

__all__ = ["build_llm_context_records", "write_llm_jsonl"]
