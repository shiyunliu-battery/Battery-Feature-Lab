# BFL: Battery Feature Lab

BFL extracts deterministic battery cycling features from standardised measurements and turns a controlled subset into evidence-backed context for AI/LLM use.

File reading, cycler detection, column mapping, unit conversion, time-axis repair and current-sign resolution are delegated to [`battery-data-standard`](https://github.com/shiyunliu-battery/battery-data-standard) (BDS), which is installed as a dependency. BFL does not reimplement ingest. Every run records the BDS conversion report in `run_metadata.json` so provenance remains auditable back to the source conversion.

The AI-facing path is deliberately one-way:

```text
cycler export
  -> battery-data-standard
  -> standardised measurements
  -> protocol/context segmentation
  -> deterministic feature tables
  -> Feature Contracts
  -> evidence records
  -> LLM context
```

`llm_context.jsonl` is generated only from contracted evidence records. Feature tables cannot bypass the evidence layer.

## Features

- Cycle summaries: capacity, energy, efficiency, C-rate, voltage, temperature, and rest time.
- Early-life curve features: `Delta Q(V)` variance, norms, quantiles, and related statistics.
- ICA/DVA features: `dQ/dV` and `dV/dQ` peaks, locations, widths, heights, and areas. AI-visible cross-cycle evidence does not assume that within-cycle peak rank identifies the same electrochemical peak across cycles.
- Relaxation features: voltage drop, slopes, interpolated voltages, and empirical exponential fits. AI-visible cross-cycle comparisons are matched by rest `step_index` and remain conditional on comparable pre-rest state/protocol.
- Stress features: voltage, current, C-rate, temperature, throughput, and SOC-dependent exposure when an explicit SOC channel is supplied. BFL does not infer SOC from voltage or generic capacity traces.
- Descriptive EIS features when impedance columns are available. The historical `EISDRTFeaturizer` import is retained for compatibility, but BFL does not perform DRT inversion or emit DRT peak placeholders.
- Observation-level trend/stress signals. The default pipeline does not assign automatic LLI/LAM/root-cause mechanism labels.
- Protocol-aware segmentation for CC/CV, pulse characterization, rest, and dynamic loads.
- Feature Contracts for every retained deterministic feature-table column; only explicitly whitelisted contracts are AI-visible.
- Evidence candidates and selected evidence packs for question-aware LLM grounding.

## Safety and interpretation policy

- Missing SOC remains not computable; no voltage-to-SOC or capacity-to-SOC surrogate is generated.
- Adding a numeric feature column does not make it AI-visible. Evidence generation requires an explicit `ai_visible=true` Feature Contract.
- Mechanism hypotheses and root-cause diagnosis are outside the default evidence layer.
- ICA peak rank is not treated as a cross-cycle peak identity.
- DRT is not claimed unless a future dedicated inversion implementation is added and contracted.

## Installation

From PyPI. This pulls in `battery-data-standard`, which BFL uses for all file reading:

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

Diagnostic/evidence thresholds remain configurable:

```bash
bfl extract input.csv \
  --output-dir out \
  --nominal-capacity-ah 1.1 \
  --datasheet-max-discharge-c-rate 5 \
  --high-soc-rest-threshold 0.25 \
  --evidence-question "Why did capacity fade after cycle 80?" \
  --evidence-token-budget 800
```

## Input Data

Ingest is handled by BDS, so BFL reads any format the installed BDS version supports. Run `bds formats` to see the current list and `bds doctor <file>` when a file will not read. JSON and JSONL are staged to CSV and then handed to BDS so there remains one normalisation path.

BDS returns its canonical schema. BFL renames it to the working names used by the featurizers:

```text
time_s, voltage_v, current_a, temperature_c, charge_capacity_ah,
discharge_capacity_ah, cycle_index, step_index, step_type
```

`cell_id` and `step_type` are not part of the BDS schema. BFL adds them, inferring `step_type` from current sign and `cell_id` from the file name when the source does not carry them. Columns BDS does not model, including explicit `soc` and the EIS triple below, are recovered from BDS pass-through fields.

Optional EIS columns are:

```text
frequency_hz, z_real_ohm, z_imag_ohm
```

An SOC column is optional. If it is absent, SOC-dependent features such as `high_soc_rest_fraction` are `NaN`/not computable and are not exposed as evidence.

## Outputs

BFL writes non-empty tables to the selected output directory. Typical outputs are:

```text
out/
  normalized_timeseries.parquet
  cycle_features.parquet
  delta_q_features.parquet
  ica_dva_features.parquet
  relaxation_features.parquet
  stress_features.parquet
  eis_features.parquet               # only when EIS inputs exist
  degradation_tags.parquet           # observation/stress signals only
  protocol_segments.parquet
  protocol_segments.jsonl
  evidence_candidates.parquet
  selected_evidence.parquet
  evidence_candidates.jsonl
  selected_evidence.jsonl
  feature_contracts.json
  llm_context.jsonl
  run_metadata.json
```

Output roles:

- `feature_contracts.json`: complete contract catalogue for retained deterministic feature-table columns. Each contract includes `definition`, `unit`, `inputs`, `method`, `method_version`, parameters, `applicability`, `quality`, `source_interval`, `interpretation_level`, and `ai_visible`. Non-whitelisted retained features are documented with `ai_visible=false` rather than silently exposed.
- `evidence_candidates.parquet` / `.jsonl`: controlled evidence objects created only from explicit AI-visible contracts, with source cycle/step/time bounds, method/version/parameters, applicability and quality status, protocol context, and interpretation level.
- `selected_evidence.parquet` / `.jsonl`: question-aware compact evidence pack under the configured token budget and redundancy constraints.
- `llm_context.jsonl`: context built only from contracted evidence records (selected evidence when available). The historical `export.llm_json_writer` module is now only a compatibility import to this evidence-only writer.
- `run_metadata.json`: input/output paths, BDS conversion report, analysis settings, contract path, and the LLM evidence-chain policy.
- `degradation_tags.parquet`: observation/stress signals such as a statistically significant decreasing capacity trend or specification-relative load flag. It does not contain automatic LLI/LAM mechanism assignments.
- `protocol_segments.parquet` / `.jsonl`: primitive test steps, conservative protocol classification, structural signatures, confidence, and matching rationale.
- `cycle_features.parquet`: per-cycle capacity, energy, efficiency, voltage/current, C-rate, and duration summaries.
- `delta_q_features.parquet`: voltage-window Delta-Q comparison features between reference and target cycles.
- `ica_dva_features.parquet`: ICA/DVA curve statistics and within-cycle peak descriptors.
- `relaxation_features.parquet`: rest-voltage recovery, slope, interpolation, and empirical fit descriptors.
- `stress_features.parquet`: usage/exposure summaries; SOC-dependent fields require an explicit SOC source.
- `eis_features.parquet`: descriptive impedance-curve features only; no DRT inversion is implied.
- `normalized_timeseries.parquet`: standardised time-series data used to compute the features.

## Feature Contract policy

AI visibility is an explicit whitelist, not a side effect of being numeric.

A Feature Contract carries:

```text
contract_id
definition
unit
inputs
method
method_version
parameter_names / resolved parameters
applicability
quality
source_interval
interpretation_level
ai_visible
```

The generated catalogue also records per-table coverage so retained feature columns can be checked against the contract registry. Generated documentation contracts default to `ai_visible=false`; promotion to AI-visible evidence requires an intentional, reviewed explicit contract.

Current AI-visible interpretation levels are:

- `derived`: deterministic descriptors computed from measurements.
- `observation`: statistically detected trends or configured stress/specification flags.

## Validation

Run the test suite:

```bash
python -m pytest
python -m ruff check .
```

Run the offline synthetic self-test:

```bash
python scripts/validate_on_dataset.py --synthetic 12
```

Run validation on a folder of per-cell CSV files:

```bash
python scripts/validate_on_dataset.py --data-dir path/to/cells --nominal-capacity-ah 1.1
```

The synthetic harness is a software self-test, not a scientific validation result. Real cross-dataset scientific validation remains a separate requirement.

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
