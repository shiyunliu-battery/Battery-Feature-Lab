# Battery Feature Lab

Battery Feature Lab extracts physically meaningful and statistically useful features from
BDS-style battery cycling exports. It is designed for downstream SOH/RUL modeling,
feature selection, SHAP analysis, and LLM-ready domain summaries.

The implementation is inspired by public battery analysis projects such as BatteryML, BEEP,
cellpy, DiffCapAnalyzer, and published early-life battery prediction work, but the code here
is written as an independent pipeline.

## What It Extracts

- Cycle summary features: capacity, energy, efficiency, C-rate, voltage, temperature, rest time.
- Early-life curve features: `Delta Q(V)` statistics such as variance, minimum, norms, quantiles.
- ICA/DVA features: `dQ/dV` and `dV/dQ` curves, peak locations, heights, widths, areas.
- Relaxation features: voltage drop, slopes, moments, interpolated voltages, exponential fits.
- Stress features: SOC, voltage, current, C-rate and temperature histograms, high-SOC rest fraction.
- EIS/DRT feature placeholders: schema and interfaces for impedance-derived features.
- Domain tags: rule-based degradation evidence for LLI, LAM_PE, LAM_NE and resistance growth.

## Install

```bash
python -m pip install -e ".[dev]"
```

## CLI

```bash
battery-features extract input.csv --output-dir out --cell-id cell_001 --nominal-capacity-ah 1.1
```

Diagnostic thresholds are intentionally configurable. For example:

```bash
battery-features extract input.csv \
  --output-dir out \
  --nominal-capacity-ah 1.1 \
  --datasheet-max-discharge-c-rate 5 \
  --high-soc-rest-threshold 0.25
```

By default, monotonic degradation tags use a Mann-Kendall trend test plus Sen's slope.
Batch stress tags use configured thresholds when supplied, otherwise batch-relative
percentiles or z-score thresholds where enough cells are available.

Expected input can use common cycler/BDS column names. The reader normalizes aliases to:

```text
time_s, voltage_v, current_a, temperature_c, charge_capacity_ah,
discharge_capacity_ah, cycle_index, step_index, step_type
```

## Outputs

```text
out/
  normalized_timeseries.parquet
  cycle_features.parquet
  delta_q_features.parquet
  ica_dva_features.parquet
  relaxation_features.parquet
  stress_features.parquet
  degradation_tags.parquet
  llm_context.jsonl
  run_metadata.json
```

## Validation

Empirically check that early-life features track cycle life (Severson et al. criterion):

```bash
# Offline self-test (no download): confirms the harness recovers log(var ΔQ(V)) vs log(life).
python scripts/validate_on_dataset.py --synthetic 12

# Real data: a folder of per-cell CSVs (MATR/Severson, BEEP exports, or BDS exports).
python scripts/validate_on_dataset.py --data-dir path/to/cells --nominal-capacity-ah 1.1
```

## Documentation

See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the plan, completed work,
and source inspirations, and [docs/FEATURE_RESEARCH.md](docs/FEATURE_RESEARCH.md) for the
literature-grounded methodology, the no-hard-code design principles, and open robustness notes.
