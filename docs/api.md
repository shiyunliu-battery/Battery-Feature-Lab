# API reference

Battery Feature Lab exposes one Python function, one result type, and one CLI command.

## Install

Python 3.11 and 3.12 are supported.

```console
python -m pip install "battery-feature-lab>=0.4,<0.5"
```

Optional extras:

```console
python -m pip install "battery-feature-lab[bdf]>=0.4,<0.5"  # native formal-BDF input
python -m pip install "battery-feature-lab[ai]>=0.4,<0.5"   # OpenAI example notebook
```

For repository development, clone the project and run `uv sync --extra dev`.

## `bfl.analyze`

```python
bfl.analyze(
    input_path,
    output_dir="bfl_outputs",
    *,
    input_adapter="auto",
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
    analysis_policy=None,
) -> AnalysisResult
```

Runs preprocessing, capability-gated analysis, schema validation, numerical recomputation, and the
six-file write contract.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `input_path` | `str | Path` | Raw cycler export or native formal-BDF artifact. |
| `output_dir` | `str | Path` | Clean run directory. BFL refuses unrelated/stale files. |
| `input_adapter` | `str` | `auto`, `bds`, or `bdf`. Prefer an explicit adapter in production. |
| `cell_id` | `str | None` | Explicit downstream identity; otherwise source stem. |
| `nominal_capacity_ah` | `float | None` | Positive declared capacity reference. Enables C-rate/SOC-dependent paths only when other gates pass. |
| `representative_cycle` | `int | None` | Explicit source cycle for cycle-scoped diagnostics. |
| `declared_protocol_name` | `str | None` | Caller declaration. BFL never infers a named protocol. |
| `formation_cycles_to_exclude` | `int` | Number of earliest source cycle IDs excluded before evolution filtering. |
| `reference_window_size` | `int` | Maximum complete comparable cycles in the capacity reference median. |
| `pulse_resistance_times_s` | `tuple[float, ...]` | Positive PyProBE pulse response times. |
| `relaxation_checkpoints_s` | `tuple[float, ...]` | Positive, unique rest-response checkpoints. |
| `voltage_column` | `str | None` | Exact provider-output column selected as analysis voltage. Does not alter normalized data. |
| `temperature_column` | `str | None` | Exact provider-output column selected as analysis temperature. Does not alter normalized data. |
| `analysis_policy` | `Mapping[str, float] | None` | Named overrides for the versioned `bfl.analysis-policy/0.1` defaults. |

Non-finite numbers, invalid ranges, empty channel names, unknown adapters, and unknown policy keys fail
before analysis. Effective configuration is stored in `analysis_evidence.json` and contributes to the
deterministic run ID; the output directory does not.

### Example: raw file through BDS

```python
import bfl

result = bfl.analyze(
    "measurements.xlsx",
    output_dir="bfl_outputs/measurements",
    input_adapter="bds",
    temperature_column="raw:Surface_Temp(degC)",
)
```

BDS is called with warning-only repair/time-sampling policies. Its native report is saved unchanged.
The generated Parquet is recorded as legacy BDF-style output, not formal BDF certification.

### Example: native formal BDF

```python
result = bfl.analyze(
    "measurements.bdf.parquet",
    output_dir="bfl_outputs/measurements",
    input_adapter="bdf",
)
```

This requires `battery-feature-lab[bdf]`. It uses the released `batterydf` read/validate API, does not
call BDS, does not reinterpret the charge-positive current convention, and writes the provider's
validation result.

### Progressive capability

Time and current are the minimum analyzable measurements. Voltage, temperature, trustworthy cycle
identity, and nominal capacity add records independently. Missing optional inputs produce
`not_computable` metrics and `not_invoked` provider calls; BFL does not fabricate columns or a
sampling frequency.

| Available capability | Added analysis |
|---|---|
| time + current | phases, current-shape modes, duration-weighted exposure, Ah throughput, current-squared exposure |
| + voltage | power/energy, current-step response, paired profiles, eligible PyProBE analyses |
| + temperature | thermal envelope and temperature-conditioned response |
| + source/joined cycle ID | structurally gated cycle/evolution analysis |
| + capacity reference | C-rate and grounded SOC-dependent provider paths |

## `AnalysisResult`

The immutable result exposes:

```python
result.output_dir
result.normalized_data_path
result.input_report_path
result.analysis_metadata_path
result.analysis_results_path
result.analysis_evidence_path
result.analysis_validation_path
result.records
result.files
```

Adapter-specific convenience properties are `bds_conversion_report_path` and
`bdf_validation_report_path`; exactly one is non-null.

## Six output artifacts

```text
normalized_data.bdf.parquet
bds_conversion_report.json       # BDS route
# or bdf_validation_report.json  # formal-BDF route
analysis_metadata.json
analysis_results.json
analysis_evidence.json
analysis_validation.json
```

- `analysis_results.json` is the compact `bfl.summary/0.1` entry point grouped by Operation,
  Response, and Evolution.
- `analysis_evidence.json` is the complete `bfl.analysis/0.1` record set with source intervals,
  methods, parameters, applicability, quality, interpretation limits, and full curve series.
- `analysis_metadata.json` is the `bfl.metadata/0.1` cell/test/dataset/channel context.
- `analysis_validation.json` is the `bfl.validation/0.1` provider-call ledger, capability matrix,
  schema result, artifact hashes, row accounting, and numerical recomputation.

The JSON contract never serializes host-local absolute paths. Source artifacts are identified by
portable filenames and SHA-256 digests, and `configuration.output_dir` is `.`. Actual filesystem
locations remain available through the `AnalysisResult` properties above.

Retrieve detailed evidence by `record_id`:

```python
import json

summary = json.loads(result.analysis_results_path.read_text(encoding="utf-8"))
evidence = json.loads(result.analysis_evidence_path.read_text(encoding="utf-8"))

compact = summary["dimensions"]["response"][0]
detailed = next(
    record
    for record in evidence["records"]
    if record["record_id"] == compact["evidence"]["record_id"]
)
```

Each scalar metric uses `{value, unit, status, reason}`. A null value cannot have `status="ok"`.
Before interpreting a record, check `applicability`, `quality`, `reference_frame`, and
`interpretation_limits`.

## CLI

```console
bfl analyze INPUT_PATH [OPTIONS]
```

Common options:

```text
--output-dir PATH
--input-adapter {auto,bds,bdf}
--cell-id TEXT
--nominal-capacity-ah FLOAT
--representative-cycle INTEGER
--declared-protocol-name TEXT
--voltage-column TEXT
--temperature-column TEXT
--formation-cycles-to-exclude INTEGER
--reference-window-size INTEGER
--pulse-resistance-time-s FLOAT       # repeatable
--relaxation-checkpoint-s FLOAT       # repeatable
```

Example:

```console
bfl analyze cell.xlsx \
  --output-dir bfl_outputs/cell \
  --input-adapter bds \
  --nominal-capacity-ah 4.0
```

The CLI prints the six generated paths on success.

## Record families

The detailed contract may contain:

```text
operation.phase_segment
operation.mode_segment
operation.window_summary
operation.exposure_summary
response.cycle_summary
response.rest_and_thermal
response.relaxation_signature
response.directional_energy_summary
response.capacity_aligned_profile
response.current_step
response.current_step_summary
response.pulse_resistance
response.ica_curve
response.dva_curve
evolution.capacity
```

Operation mode vocabulary is closed: `constant_current_like`, `constant_voltage_like`, `pulse_like`,
`dynamic_current`, and `unmatched`.

## Scientific limits exposed by the API

- Local current-step resistance is an apparent terminal `delta-V/delta-I`, not intrinsic resistance
  or SOH.
- Relaxation is finite terminal-voltage recovery, not equilibrium OCV.
- Capacity-aligned coordinates are not SOC unless explicitly grounded.
- Evolution requires source/joined cycle identity, structural completeness, and comparable operation.
- The API does not return a named protocol, chemistry inference, safety diagnosis, ageing conclusion,
  or natural-language report.
