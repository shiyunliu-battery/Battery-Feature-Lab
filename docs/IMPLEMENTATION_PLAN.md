# Implementation Plan and Status

## Goal

Build a complete Python 3.10+ feature extraction pipeline for BDS-exported battery datasets.
The pipeline converts raw cycling time series into structured feature tables and LLM-ready
JSON records for battery raw-data analysis.

## Source Inspirations

The implementation references feature ideas and architecture patterns from public work:

- BatteryML: unified battery data preprocessing, feature extraction, modeling interfaces.
- BEEP: cycler parsing, structured summaries, early prediction workflow.
- Severson et al. Nature Energy 2019: early-life `Delta Q(V)` features.
- cellpy: common battery cycling data abstractions and ICA/relaxation analysis.
- DiffCapAnalyzer: quantitative differential capacity peak descriptors.
- TUM vehicle diagnostics repositories: DVA/ICA extraction, filtering, pack/vehicle thinking.
- PyBaMM: future physics-informed validation and synthetic degradation studies.

No code is copied from these repositories. Feature definitions and module boundaries are
implemented independently for this project.

## Completed

- [x] Project package structure.
- [x] BDS-style CSV/JSON/Parquet reader with column alias normalization.
- [x] Data validation and unit/sign normalization utilities.
- [x] Cycle splitting and step-type inference.
- [x] Cycle summary feature extraction.
- [x] Early-life `Delta Q(V)` feature extraction.
- [x] ICA/DVA curve generation and peak feature extraction.
- [x] Relaxation/rest feature extraction.
- [x] Stress histogram and dynamic-operation feature extraction.
- [x] EIS/DRT feature extraction interfaces and robust placeholders.
- [x] Rule-based degradation-domain tagging.
- [x] Hard-coded diagnostic thresholds replaced with configurable/statistical criteria.
- [x] Feature selection utilities.
- [x] Optional SHAP report adapter.
- [x] Parquet and LLM JSONL exports.
- [x] CLI entry point.
- [x] Synthetic tests covering core feature extraction.

## Planned Next

- [ ] Add native BDS API client once export schema is confirmed.
- [ ] Add richer chemistry-specific ICA/DVA degradation maps.
- [ ] Add BatteryML/BEEP dataset adapters for benchmark validation.
- [ ] Add model training recipes for SOH, RUL, EOL, anomaly detection.
- [ ] Add a report generator with SHAP plots and feature drift visualizations.
- [ ] Add a lightweight web UI for inspecting cycles and feature evidence.

## Pipeline Stages

1. **Input adapter**
   - Read BDS CSV/JSON/Parquet.
   - Normalize columns to canonical names.
   - Validate required voltage/current/time fields.
   - Infer cycle and step types if missing.

2. **Core processing**
   - Sort by cell, cycle, time.
   - Normalize current sign convention.
   - Integrate capacity/energy where capacity columns are missing.
   - Resample curves onto stable voltage/capacity grids.
   - Smooth before numerical differentiation.

3. **Feature extraction**
   - `cycle_summary`: capacity, energy, efficiency, temperature, voltage, C-rate.
   - `delta_q`: early cycle `Q(V)` difference statistics.
   - `ica_dva`: `dQ/dV` and `dV/dQ` peaks and curve statistics.
   - `relaxation`: rest voltage, slopes, moments, exponential fit parameters.
   - `stress`: SOC/current/temperature/voltage histograms and duty-cycle metrics.
   - `eis_drt`: impedance/DRT schema for future EIS imports.

4. **Analysis**
   - Correlation, mutual information, variance filtering, VIF.
   - Optional SHAP explanations when compatible packages are installed.
   - Rule-based degradation tags for LLM context.
   - Trend tags use Mann-Kendall p-values and Sen's slope instead of fixed slope cutoffs.
   - Stress tags use configured thresholds or batch-relative percentiles/z-scores.
   - Datasheet-dependent C-rate warnings require a supplied datasheet limit.

## Diagnostic Threshold Policy

Release-facing diagnostic tags must avoid undocumented absolute thresholds. Current policy:

- Capacity fade: Mann-Kendall trend test on normalized discharge capacity; tag only when
  `p < trend_p_value_alpha` and Sen's slope is negative.
- ICA/DVA drift: Mann-Kendall trend test on primary peak position; report p-value and
  Sen's slope in evidence text.
- Relaxation tau increase: Mann-Kendall trend test over all available rest segments, not
  a first-vs-last ratio.
- High-SOC rest: use `high_soc_rest_fraction_threshold` when provided, otherwise use a
  batch percentile when at least two cells are available.
- Dynamic current variance: use C-rate variance, not raw A^2, with configured threshold
  or batch z-score threshold.
- High instantaneous discharge: only tag when `datasheet_max_discharge_c_rate` is provided.
- Delta-Q minimum points: uses `FeatureConfig.min_points_for_curve`.
- ICA/DVA peak prominence: uses MAD-estimated noise times
  `FeatureConfig.peak_prominence_noise_multiplier`.

5. **Export**
   - Parquet feature tables for modeling.
   - JSONL context records for LLM workflows.
   - Run metadata for reproducibility.
