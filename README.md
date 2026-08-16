# Battery Feature Lab 0.4

Battery Feature Lab (BFL) converts supported battery-cycler exports into normalized data and auditable JSON analysis records for downstream software.

The public Python API is intentionally small:

```python
import bfl

result = bfl.analyze(
    "cell.csv",
    output_dir="bfl_outputs",
    input_adapter="auto",  # auto | bds | bdf
    cell_id=None,
    nominal_capacity_ah=None,
    representative_cycle=None,
    declared_protocol_name=None,
    formation_cycles_to_exclude=1,
    reference_window_size=4,
    pulse_resistance_times_s=(10.0,),
    relaxation_checkpoints_s=(10.0, 30.0, 60.0, 300.0, 600.0, 1800.0),
    voltage_column=None,
    temperature_column=None,
    analysis_policy=None,  # optional named threshold overrides
)
```

The equivalent CLI is:

```console
bfl analyze cell.csv --output-dir bfl_outputs --nominal-capacity-ah 1.1
```

Every successful run writes six machine-readable files. The provider-native
input report depends on the selected adapter:

```text
normalized_data.bdf.parquet
bds_conversion_report.json        # raw/BDS handoff
# or bdf_validation_report.json   # native formal-BDF handoff
analysis_metadata.json
analysis_results.json
analysis_evidence.json
analysis_validation.json
```

`analysis_results.json` is the compact `bfl.summary/0.1` entry point for downstream software. `analysis_evidence.json` keeps the complete `bfl.analysis/0.1` records, source intervals, methods and curve arrays. `analysis_metadata.json` stores tool-grounded cell, test, dataset and channel context. BFL does not generate prose reports or rendered text.

## Tool responsibilities

- BFL analysis consumes one provider-neutral, preprocessed table. `input_adapter="auto"` routes raw cycler files through `battery-data-standard>=0.3.1,<0.4` and explicitly identified BDF artifacts through formal BDF. Ambiguous files can be forced with `input_adapter="bds"` or `input_adapter="bdf"`; a provider error never triggers a silent fallback to the other adapter.
- The BDS path performs vendor conversion, unit/current-sign normalization and source tracking with `repair_policy="warn"` and `time_sampling_policy="warn"`. Its native `ConversionReport.write_json()` output is preserved verbatim. BDS `target="bdf"` is recorded as legacy BDF-style column compatibility, not formal BDF certification.
- The optional `battery-feature-lab[bdf]` extra pins `batterydf==0.1.0` and uses its released `read` and `validate` APIs for native BDF CSV/Parquet artifacts. This path never passes data through BDS or rewrites the BDF charge-positive current convention. It writes the provider-native `bdf_validation_report.json`; a passed validator is reported as such, not as certification of an ontology version.
- `PyProBE-Data>=2.6,<2.7` provides cycling summaries, pulse resistance, and LEAN differentiation for ICA/DVA. Provider failures are recorded as `provider_error`; BFL does not silently substitute a same-name local algorithm.
- SciPy provides peak finding, Theil–Sen slope, and Kendall tau-b.
- BFL supplies only the project-specific rules and numerical conventions that these tools do not expose: source gating, structural completeness, conservative operation labels, previous-sample ZOH integration, duration-weighted exposure, and comparable-cycle selection.
- Metadata is taken first from the selected preprocessing provider, adjacent BDF sidecar references, and explicit user declarations. BattINFO supplies a vocabulary reference and Battery Data Toolkit supplies a metadata field-model reference; neither is used to invent missing cell facts. PyProBE bridge objects do not replace source metadata.

Time and current are the minimum analyzable capability. With those channels BFL
still produces phase, current-shape mode, duration-weighted current exposure,
Ah throughput and current-squared exposure. Voltage enables power, energy,
current-step response, paired profiles and eligible PyProBE analyses;
temperature independently enriches operating/thermal context. Missing optional
channels produce per-feature `not_computable` records and `not_invoked`
provider calls rather than fabricated columns or a failed whole run. A literal
current-only array without a measured time coordinate is not integrated.

The supported runtime is Python 3.11 or 3.12.

## Analysis records

Each record contains stable identity and scope fields, source row/time intervals, attributes, scalar metrics, optional curve series, exact method provenance, applicability, quality flags, and interpretation limits.

Record types are:

- `operation.phase_segment`
- `operation.mode_segment`
- `operation.window_summary`
- `operation.exposure_summary`
- `response.cycle_summary`
- `response.rest_and_thermal`
- `response.relaxation_signature`
- `response.directional_energy_summary`
- `response.capacity_aligned_profile`
- `response.current_step`
- `response.current_step_summary`
- `response.pulse_resistance`
- `response.ica_curve`
- `response.dva_curve`
- `evolution.capacity`

Operation mode labels are deliberately limited to `constant_current_like`, `constant_voltage_like`, `pulse_like`, `dynamic_current`, and `unmatched`. BFL never infers a named test protocol.

The compact downstream view is organized into three analysis dimensions, while metadata and provenance apply to every result:

- `operation.window_summary` describes duration, phase/mode fractions, current and power exposure, throughput, voltage/temperature envelope, and the dominant observed operating mode.
- `response.directional_energy_summary` describes charge/discharge throughput, directional energy and mean voltage using previous-sample ZOH. Its balanced-window ratio is explicitly not called cycle efficiency.
- `response.current_step` and `response.current_step_summary` describe rest-referenced terminal `delta-V/delta-I` at fixed response times. They do not infer SOC, intrinsic resistance, SOH, or a named pulse protocol. PyProBE `response.pulse_resistance` remains a separate, stricter path that requires a grounded capacity reference.
- `response.capacity_aligned_profile` emits charge/discharge voltage curves only when state-window, capacity and current comparability gates pass. Standardized temperature is also matched when available; if it is absent, the record is warned and cannot support thermal or cross-test comparison. Its capacity coordinate is explicitly not SOC.
- `response.relaxation_signature` aggregates terminal-voltage recovery at the configured checkpoints (10 s to 1800 s by default) and compact shape descriptors after observed charge or discharge phases. It does not claim equilibrium OCV or a degradation mechanism.
- `evolution.capacity` describes comparable repeated cycle observations only when source or joined cycle identifiers and completeness gates permit it; absence of those inputs produces an explicit unavailable result.

These records report source intervals, reference frames, deterministic quality gates, categorical confidence (never a probability), method parameters, and interpretation limits. Summaries with candidate gates also report rejection counts by reason. Capacity-aligned charge/discharge profiles are intentionally omitted unless a complete comparable phase pair is available; BFL does not emit a placeholder curve for incomplete data.

## Numerical conventions

Irregularly sampled exposure and fallback capacity/energy integration use a left-continuous previous-sample hold: `x[i]` applies on `[t[i], t[i+1])`. Missing held values are excluded, not replaced with zero. The final sample has zero duration. Sampling QA flags only an isolated interval that exceeds both neighbouring positive intervals by the configured factor; it calls this a `sampling_interval_outlier`, not a missing-data gap. Sustained multi-rate logging blocks therefore remain valid. Outlier intervals remain included in ZOH totals and their count, duration, and maximum are reported.

Reported cumulative capacity or energy columns are used only when they contain at least two finite points and are monotonic with a non-negative endpoint delta. Otherwise BFL uses the declared `zoh_previous_v1` branch and records why the reported column was unusable.

Capacity evolution accepts only source or joined cycle identifiers, structurally complete cycles, and one conservative operation signature. The default reference is the median of up to four complete cycles after excluding one formation cycle, with at least three reference cycles required. Theil–Sen and Kendall tau-b are emitted only with at least eight comparable cycles; no trend p-value or significance claim is produced.

Scientific defaults are held in the versioned `bfl.analysis-policy/0.1`
registry, not inferred from a filename, step number, row location, or the
example dataset. Python callers may override named entries through
`analysis_policy={...}`. Unknown names and invalid ranges are rejected. The
complete resolved policy and its version are written to
`analysis_evidence.json.configuration`; each method records its relevant
effective parameters, so a threshold change also changes the run identity. Relaxation checkpoint times are
configured separately with `relaxation_checkpoints_s`. Adding voltage or
temperature enriches the available records without changing already-computed
time/current metrics.

## Reading the outputs

```python
import json
from pathlib import Path

output_dir = Path("bfl_outputs")
results = json.loads((output_dir / "analysis_results.json").read_text())
metadata = json.loads((output_dir / "analysis_metadata.json").read_text())
evidence = json.loads((output_dir / "analysis_evidence.json").read_text())
validation = json.loads((output_dir / "analysis_validation.json").read_text())

indexed = results["dimensions"]["response"][0]
record = next(
    item for item in evidence["records"] if item["record_id"] == indexed["evidence"]["record_id"]
)
print(validation["status"], metadata["cell"], record["source_intervals"])
```

`analysis_validation.json` links all six artifacts, records the input adapter,
handoff status, channel capability matrix, software versions and every
third-party call, verifies compact-index evidence references, and reports
schema validation and numerical recomputation without duplicating the native
input report.

## Development

```console
uv sync --python 3.12 --extra dev
# Add --extra bdf when native formal-BDF input is required.
uv run pytest -q
uv run ruff check .
```

The tutorial notebook is [examples/BFL_example.ipynb](examples/BFL_example.ipynb). Raw validation datasets belong in `tests/data/real/` and are intentionally ignored; only their manifest and usage notes are versioned.
