# API reference

Battery Feature Lab exposes `analyze()` for time-series measurements and
`analyze_eis()` for spectra, with an `AnalysisResult` output object. The EIS
interface and its modality-specific contract are documented in [EIS analysis](eis.md).

## Install

Python 3.11 and 3.12 are supported.

```console
python -m pip install "battery-feature-lab>=0.7,<0.8"
```

Optional extras:

```console
python -m pip install "battery-feature-lab[bdf]>=0.7,<0.8"  # native formal-BDF input
python -m pip install "battery-feature-lab[ai]>=0.7,<0.8"   # OpenAI example notebook
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
| `analysis_policy` | `Mapping[str, float] | None` | Named overrides for the versioned `bfl.analysis-policy/0.3` defaults. |

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

### Phase segment records

Each `operation.phase_segment` record (method `source_step_and_current_phase_segmentation_v2`)
covers one source step or one run of a phase. Besides `duration` and `sample_count` it gives, from
the segment's own samples:

| Metric | Unit | Meaning |
|---|---|---|
| `start_voltage`, `end_voltage`, `voltage_min`, `voltage_max` | V | First and last finite voltage and the range. |
| `absolute_current_median`, `absolute_current_min`, `absolute_current_max` | A | Statistics of the current magnitude. |
| `zoh_capacity`, `zoh_energy` | Ah, Wh | Current magnitude and power magnitude by previous-sample ZOH over the segment's support: the last sample is held until the next segment's first sample, as `duration` is measured. |
| `trapezoid_capacity`, `trapezoid_energy` | Ah, Wh | The same integrands by the trapezoid rule between the first and last samples of the segment. |
| `reported_capacity_start`, `reported_capacity_end`, `reported_energy_start`, `reported_energy_end` | Ah, Wh | The source's own accumulator at the first and last finite samples: a directional column for the segment's phase when present (`charge_capacity_ah`, `discharge_capacity_ah`, ...), otherwise the undirected column the source wrote (`source_capacity_ah`, `source_energy_wh`). The column used is in `attributes.observation_columns`. |
| `reported_capacity_change`, `reported_energy_change` | Ah, Wh | Last minus first finite value, only when the values do not decrease inside the segment; otherwise `not_computable` (`reported_column_not_monotonic`). It does not depend on where the source resets its accumulator. |
| `trapezoid_capacity_minus_reported_change`, `trapezoid_energy_minus_reported_change` | Ah, Wh | Trapezoid integral between the same two samples minus the reported change; positive when the integral exceeds the source's value. |
| `start_temperature`, `end_temperature`, `temperature_min`, `temperature_max`, `temperature_mean` | degC | The analysis temperature channel. |
| `start_ambient_temperature`, ..., `ambient_temperature_mean` | degC | The ambient channel, when it differs from the analysis channel. |

A missing channel gives `not_computable` (`channel_absent` or `no_finite_samples`), never an
estimate. Temperature statistics use every finite sample; screening for acquisition artefacts is
left to the intake (BDS reports abrupt temperature excursions with their record intervals).

The reconciliation uses the trapezoid rule because the source integrates far more often than it
logs: between two logged samples of a smooth single-phase signal the trapezoid's error is second
order in the interval and ZOH's first order. On the Chen et al. (2020) LG M50 files, ZOH energy of a
1.5C discharge logged every 16 s exceeds the tester's energy by about 0.14 %; the trapezoid agrees
with it within 1 mWh.

### Capacity–voltage curves

Each charge or discharge phase segment also gets a `response.capacity_voltage_curve` record (method
`trapezoid_capacity_voltage_polyline_v1`) linked to it by `attributes.phase_segment_record_id`. Its
`series` holds `capacity_ah` (capacity moved since the segment's first record, trapezoid rule on the
current magnitude over test time; one sample per capacity value, the last) and `voltage_v`.

The samples are thinned by a vertical-deviation Ramer–Douglas–Peucker pass. A sample is kept only where
dropping it would move the linear interpolation by more than `curve_voltage_tolerance_v` (policy, default
0.5 mV). Both polylines break only at samples, so the bound checked at every sample holds everywhere. Two
curves can then be compared on any capacity grid with an error of at most the sum of their tolerances.

| Metric | Unit | Meaning |
|---|---|---|
| `capacity` | Ah | Total capacity of the segment; equals the phase segment's `trapezoid_capacity`. |
| `point_count`, `distinct_capacity_samples` | 1 | Points kept, and samples with distinct capacity. |
| `voltage_tolerance` | V | The bound that holds. It is the policy value unless more than `curve_max_points` (default 1000) points would be kept; then it is doubled until they are not, and the record carries the flag `curve_tolerance_relaxed_to_point_limit`. |
| `max_interpolation_error` | V | The largest deviation found; at most `voltage_tolerance`. |

A segment with fewer than two distinct capacity values is `not_computable`. Capacity is not a state of
charge, and the curve adds nothing between samples.

On the Chen et al. (2020) LG M50 files, 0.5 mV keeps 65–90 points of the 146–1214 distinct samples of each full discharge, and 17–19 of about 60 for the initial partial discharge. Cell-to-cell
voltage RMSE computed from the stored curves differs from the RMSE computed from the raw samples by at most
0.08 mV, and the maximum voltage difference by at most 0.32 mV. A uniform 1001-point capacity grid misses
the end-of-discharge knee by up to 11 mV.

### Rest records

Each `response.rest_and_thermal` record (method `rest_filter_with_interval_descriptors_v2`) gives,
besides the checkpoint voltages and the thermal descriptors:

| Metric | Unit | Meaning |
|---|---|---|
| `preceding_load_end_voltage` | V | Last finite voltage of the charge or discharge segment immediately before the rest. |
| `end_drift_rate` | V/h | Least-squares slope of voltage on test time over the final `relaxation_end_window_s` (policy, default 600 s) of the rest, one sample per time stamp (the last); `not_computable` below `relaxation_end_window_min_samples` (default 3). |
| `end_window_sample_count`, `end_window_span` | 1, s | Samples and time span used for the end drift. |
| `uncounted_wall_time` | s | Wall-clock time inside the rest that the test clock did not count: the sum of consecutive-sample differences between wall-clock and test-clock steps above the larger of `clock_divergence_min_s` (policy, default 120 s) and twice the wall clock's resolution. `not_computable` without a wall-clock column (`unix_time_s` or `Unix Time / s`). A positive value adds the quality flag `test_clock_pause_within_interval`. |
| `uncounted_wall_time_interval_count` | 1 | Number of such consecutive-sample intervals. |

Durations and checkpoints remain on the test clock, so a rest that contains a pause is longer in
wall-clock time than its `duration`. An end drift that is not zero means the rest ended before
equilibrium; it does not say how far from equilibrium.

## Scientific limits exposed by the API

- Local current-step resistance is an apparent terminal `delta-V/delta-I`, not intrinsic resistance
  or SOH.
- Relaxation is finite terminal-voltage recovery, not equilibrium OCV.
- Capacity-aligned coordinates are not SOC unless explicitly grounded.
- Evolution requires source/joined cycle identity, structural completeness, and comparable operation:
  the same phase signature and a median discharge current within
  `evolution_discharge_current_relative_tolerance` (default 5 %), so discharges at different
  rates are never treated as repeats. A discharge-temperature spread above
  `evolution_temperature_difference_max_deg_c` (default 2 °C) is flagged.
- The API does not return a named protocol, chemistry inference, safety diagnosis, ageing conclusion,
  or natural-language report.
