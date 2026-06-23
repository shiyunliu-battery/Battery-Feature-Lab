# BFL: Battery Feature Lab

BFL extracts battery cycling features from BDS-style exports and common cycler tables. It turns
raw time-series data into feature tables for SOH/RUL modeling, feature screening, explainability,
and compact diagnostic summaries.

## Features

- Cycle summaries: capacity, energy, efficiency, C-rate, voltage, temperature, and rest time.
- Early-life curve features: `Delta Q(V)` variance, norms, quantiles, and related statistics.
- ICA/DVA features: `dQ/dV` and `dV/dQ` peaks, locations, widths, heights, and areas.
- Relaxation features: voltage drop, slopes, interpolated voltages, and exponential fits.
- Stress features: SOC, voltage, current, C-rate, temperature histograms, and high-SOC rest time.
- EIS descriptors when impedance columns are available.
- Rule-based degradation tags for LLI, LAM_PE, LAM_NE, resistance growth, and related evidence.
- JSONL context summaries for reports and review.

## Installation

From PyPI:

```bash
pip install battery-feature-lab
```

From a local checkout:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

Python:

```python
from pathlib import Path
import bfl

result = bfl.extract(
    "/content/25-LFP-1.csv",
    output_dir="/content/bfl_outputs",
    nominal_capacity_ah=1.2,
    reference_cycle=2,
    target_cycle=5,
)

print(result.llm_context_path)
for path in result.files:
    print(path.name)
```

CLI:

```bash
bfl extract input.csv --output-dir out --cell-id cell_001 --nominal-capacity-ah 1.1
```

The longer command name is also available:

```bash
battery-features extract input.csv --output-dir out --cell-id cell_001 --nominal-capacity-ah 1.1
```

## Example Notebook

See [examples/BFL_example.ipynb](examples/BFL_example.ipynb) for an example that installs BFL,
runs feature extraction on a BDS CSV, and prints the exported files.

Diagnostic thresholds are configurable:

```bash
bfl extract input.csv \
  --output-dir out \
  --nominal-capacity-ah 1.1 \
  --datasheet-max-discharge-c-rate 5 \
  --high-soc-rest-threshold 0.25
```

## Input Data

BFL accepts CSV, TSV, JSON, JSONL, and Parquet files. Input data can use common cycler/BDS
column names. The reader normalizes aliases to:

```text
time_s, voltage_v, current_a, temperature_c, charge_capacity_ah,
discharge_capacity_ah, cycle_index, step_index, step_type
```

At minimum, the input should contain time, voltage, current, and enough cycle/step information
to identify charge, discharge, and rest periods. If `cell_id`, `cycle_index`, or `step_type` is
missing, BFL can infer or fill parts of the schema from the file name, current sign, and command
line options.

Optional EIS columns are:

```text
frequency_hz, z_real_ohm, z_imag_ohm
```

## Outputs

BFL writes non-empty tables to the selected output directory. The Parquet files contain the
feature tables; `llm_context.jsonl` is a compact context summary for reports and review.

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

Output roles:

- `llm_context.jsonl`: cell summary with dataset overview, data-quality warnings,
  capacity/efficiency trends, selected Delta-Q/ICA-DVA/relaxation/stress highlights, diagnostic
  evidence, cell context, analysis configuration, and reliability notes. When
  `nominal_capacity_ah` is provided, this file also includes nameplate-relative SOH fields; when
  it is not provided, the nominal-capacity fields remain `null` and BFL records an explicit
  `nominal_capacity_missing` warning. Its `data_quality.quality_summary` block reports BFL's
  automatic checks for cycle completeness, capacity reliability, reference/target cycle suitability,
  and feature computability.
- `run_metadata.json`: input path, output paths, reader settings, feature settings, and diagnostic
  settings used for the run.
- `degradation_tags.parquet`: rule-based diagnostic evidence signals and confidence labels.
- `cycle_features.parquet`: per-cycle capacity, energy, efficiency, voltage/current, C-rate, and
  duration summaries.
- `delta_q_features.parquet`: voltage-window Delta-Q comparison features between reference and
  target cycles.
- `ica_dva_features.parquet`: ICA/DVA curve statistics and peak descriptors by cycle.
- `relaxation_features.parquet`: rest-voltage recovery, slope, and exponential-fit features.
- `stress_features.parquet`: whole-cell stress exposure, SOC/voltage/C-rate histograms, throughput,
  and equivalent full cycles.
- `normalized_timeseries.parquet`: standardized time-series data used to compute the features.

## Validation

Run the offline self-test:

```bash
python scripts/validate_on_dataset.py --synthetic 12
```

Run validation on a folder of per-cell CSV files:

```bash
python scripts/validate_on_dataset.py --data-dir path/to/cells --nominal-capacity-ah 1.1
```

## Python Usage

```python
from pathlib import Path

from battery_feature_lab.pipeline import FeaturePipeline, PipelineConfig
from battery_feature_lab.schemas import ExportConfig, FeatureConfig, ReaderConfig

pipeline = FeaturePipeline(
    PipelineConfig(
        reader=ReaderConfig(cell_id="cell_001"),
        features=FeatureConfig(nominal_capacity_ah=1.1),
        export=ExportConfig(output_dir=Path("out")),
    )
)

tables = pipeline.run("input.csv")
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

## License

MIT License. See [LICENSE](LICENSE).
