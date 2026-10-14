# Changelog

## 0.7.0 — 2026-10-14

- Adopted `eisfit==0.2.0`. Parameter units now come from eisfit; a fitted parameter without
  a unit is still rejected with `provider_invalid_result:ecm_parameter_units_undeclared`.
- ECM records keep each multistart start's start and final parameters, and report
  `attributes.multistart_near_best` (also in the compact index): the starts whose cost is
  within the eisfit `near_best_rtol` of the best, and each parameter's range across them.
  BFL applies no stability threshold; parameter stability remains `not_assessed`.
- Added `near_best_rtol` to the validated ECM fit options. ECM method parameters no longer
  include figure options, which eisfit 0.2.0 moved out of its fit configuration.
- `main` is the only branch: releases are published by pushing a `v<version>` tag, and the
  GitHub release notes are the version's changelog section instead of the whole changelog.
- The Zhang et al. example outputs remain a BFL 0.6.0 / eisfit 0.1.1 snapshot.

## 0.6.0 — 2026-10-14

- Added an EIS example notebook for the CC BY 4.0 Zhang et al. (2020) dataset. It screens all
  261 state-V spectra of cell 25C01 with Lin-KK, fits and compares six of them under the
  `ageing` gate, shows the comparison gate's outcomes, and adds an optional GPT-5.6 Sol step
  over four input conditions.
- Added an examples index with dataset attribution, layout and running instructions.
- Fixed provenance: every BFL record now reads its provider version from the installed package
  metadata, for example `BFL 0.6.0`. Records produced by 0.5.0 reported `0.4.0` or `BFL 0.4.0`.
- Lin-KK records now report `frequency_of_maximum_residual`, `points_above_residual_limit` and
  `fraction_points_above_residual_limit`.
- A non-finite Lin-KK μ or invalid Lin-KK arrays are reported as
  `provider_invalid_result:lin_kk_mu_not_finite` or `provider_invalid_result:lin_kk_invalid_outputs`,
  with the termination facts retained in `attributes.lin_kk_termination`.
- EIS provider calls record captured console line counts and warnings in `provider_output`,
  with absolute paths in warning messages replaced by `<path>`.
- **Changed contract:** the EIS compact summary is now `bfl.eis-summary/0.2`. Conditions shared
  by every spectrum and interpretation-limit text are stated once at run level, comparisons name
  their spectra instead of repeating condition blocks, and empty time-series member fields are
  omitted. Evidence records are unchanged. On the Zhang et al. example, the compact file shrinks
  from about 31k to 23k tokens.
- Corrected the API reference install ranges, which pointed to 0.4.x, and listed the EIS schemas in the
  documentation index.
- Kept the exact `eisfit==0.1.1` pin and made its integration explicit: named lists of validated
  fit options and tails, a declared unit for every fitted parameter (an undeclared parameter now
  yields `provider_invalid_result:ecm_parameter_units_undeclared` instead of a dimensionless
  value), and a release reference built from the installed eisfit version.
- Kept pyarrow below 26 in the `eis` extra. pyarrow 26 requires NumPy 2 without declaring it, so a
  pip installation of the extra could pair it with the required NumPy 1.26; BFL analysis still
  ran, but importing pyarrow in that environment (for example Polars `to_pandas()` or pandas
  Parquet I/O) failed.
- Added an eisfit contract test that fits a small spectrum for each tail and checks the fields,
  shapes, units and JSON serialisation BFL depends on, and that the installed version matches
  the pin. Documented the eisfit upgrade procedure.

## 0.5.0 — 2026-10-14

- Added an independent `analyze_eis()` API and `analyze-eis` CLI.
- Added BDS-backed spectrum grouping, explicit phase units, source-row
  provenance, condition declarations and retained/excluded row accounting.
- Added versioned Lin-KK diagnostics, released `eisfit 0.1.1` fitting and
  explicit provider-failure records without numerical fallback.
- Added condition-gated repeat, ageing, SOC and temperature comparisons.
- Added modality-specific schemas, a six-artifact EIS evidence contract and
  a CC BY 4.0 real-spectrum regression fixture with attribution.
- Constrained Polars to 1.x for PyProBE cycling and differentiation compatibility.
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
