# Changelog

## 0.5.0 — 2026-10-14

- Added an independent `analyze_eis()` API and `analyze-eis` CLI.
- Added BDS-backed spectrum grouping, explicit phase units, source-row
  provenance, condition declarations and retained/excluded row accounting.
- Added versioned Lin-KK diagnostics, released `eisfit 0.1.1` fitting and
  explicit provider-failure records without numerical fallback.
- Added condition-gated repeat, ageing, SOC and temperature comparisons.
- Added modality-specific schemas, a six-artifact EIS evidence contract and
  a CC BY 4.0 real-spectrum regression fixture with attribution.
- Locked the EIS extra to NumPy 1.26 for `impedance 1.7.1` compatibility.

## 0.4.0

- Added the single `bfl.analyze()` API and `bfl analyze` CLI.
- Added independent BDS raw-conversion and optional formal-BDF input adapters over one provider-neutral analysis handoff.
- Added capability-gated time+current analysis; voltage and temperature enrich dependent records without becoming global requirements.
- Added the required PyProBE cycling, pulse-resistance, and ICA/DVA analysis path.
- Added auditable previous-sample ZOH statistics, structural completeness, and comparable-cycle evolution.
- Added compact operating-window, directional energy, phase-conditioned relaxation, and protocol-neutral current-step records for short observations.
- Added explicit current-step rejection counts, reference frames, categorical confidence, and limits that prevent health or mechanism claims from short data.
- Split the downstream contract into a compact results index, tool-grounded metadata, full retrievable evidence, validation, normalized BDF data and the native BDS report.
- Added conservative capacity-aligned voltage profiles, expanded fixed-time relaxation descriptors, explicit analysis-channel overrides and metadata/provenance references.
- Added a reproducible CC BY 4.0 Catenaro–Onori example dataset and an optional GPT-5.6 Responses API notebook step that interprets redacted BDS/BFL JSON without storing API credentials or model output.
- Licensed Battery Feature Lab 0.4.0 under MPL-2.0 while retaining the example dataset's upstream CC BY 4.0 terms.
- Replaced serialized host-local paths with portable filename-and-SHA256 artifact references.
