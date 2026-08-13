# BFL: Battery Feature Lab

Battery Feature Lab (BFL) converts standardised battery cycling measurements into deterministic feature tables and a controlled, auditable evidence layer for downstream AI/LLM use.

File reading, cycler detection, column mapping, unit conversion, time-axis repair and current-sign normalisation are delegated to `battery-data-standard` (BDS). BFL starts after that normalisation step.

## Architecture

```text
cycler export
  -> battery-data-standard
  -> standardised measurements
  -> protocol/context segmentation
  -> deterministic battery features
  -> Feature Contract whitelist
  -> evidence records
  -> LLM context
```

The LLM context writer accepts evidence records only. Numeric feature columns are never exposed automatically.

## Safety and interpretation policy

- SOC-dependent evidence requires an explicitly supplied SOC channel. Missing SOC remains missing; BFL does not synthesise SOC from voltage or generic capacity traces.
- Degradation outputs are observation/stress flags only. The default pipeline does not assign automatic root-cause mechanisms.
- The historical `EISDRTFeaturizer` import is retained for compatibility, but it now emits descriptive `eis_features` only. DRT inversion and DRT peak placeholders are not implemented or claimed.
- Unregistered numeric columns cannot enter the evidence layer.
- Every AI-visible feature must have a Feature Contract describing definition, unit, inputs, method, method version, parameters, applicability, quality requirements, source interval and interpretation level.

## Retained feature families

- Cycle summaries: capacity, energy, efficiency, duration, current, voltage and temperature.
- Early-life Delta Q(V): variance, norms and integrated curve-change descriptors.
- ICA/DVA: curve and peak descriptors remain in feature tables; AI-visible cross-cycle evidence is restricted to same-step-type area comparisons. Untracked peak-identity drift is not exposed.
- Relaxation: per-rest descriptors; cross-cycle evidence is matched by step index and marked conditional.
- Stress/history: throughput, C-rate and thermal descriptors. High-SOC rest exposure is available only when SOC is explicitly supplied.
- Protocol context: conservative CC/CV/rest/pulse/dynamic segmentation with heuristic named-protocol confidence metadata.

## Installation

```bash
pip install battery-feature-lab
```

Development:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
import bfl

result = bfl.extract(
    "cell.csv",
    output_dir="bfl_outputs",
    nominal_capacity_ah=1.1,
    reference_cycle=10,
    target_cycle=100,
)

print(result.llm_context_path)
print(result.selected_evidence_path)
```

CLI:

```bash
bfl extract input.csv --output-dir out --cell-id cell_001 --nominal-capacity-ah 1.1
```

## Outputs

Typical output directory:

```text
normalized_timeseries.parquet
cycle_features.parquet
delta_q_features.parquet
ica_dva_features.parquet
relaxation_features.parquet
stress_features.parquet
eis_features.parquet              # only when EIS columns are present
degradation_tags.parquet
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

### Feature Contracts

`feature_contracts.json` is the machine-readable registry for AI-visible features. A feature without an explicit `ai_visible=true` contract does not enter the evidence layer.

Each contract contains:

```text
definition
unit
inputs
method
method_version
parameter_names
applicability
quality
source_interval
interpretation_level
ai_visible
```

### Evidence records

Evidence records bind a contracted feature or observation to its value/unit, source cycle/step and time bounds, method/version/parameters, applicability and quality status, protocol context, and interpretation level.

### LLM context

`llm_context.jsonl` is generated only from contracted evidence records (selected evidence when available). It does not re-read feature tables to construct a second independent summary.

## Interpretation levels

Current AI-visible outputs use:

- `derived`: deterministic features computed from measurements.
- `observation`: statistically detected trends or configured stress-context flags.

Mechanism hypotheses and root-cause diagnosis are outside the default evidence layer.

## Validation

```bash
python -m pytest
python -m ruff check .
```

The synthetic validation harness is a software self-test, not scientific validation:

```bash
python scripts/validate_on_dataset.py --synthetic 12
```

Real cross-dataset scientific validation remains a separate requirement.

## License

MIT License. See [LICENSE](LICENSE).
