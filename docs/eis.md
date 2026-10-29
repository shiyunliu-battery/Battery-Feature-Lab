# EIS analysis

EIS has an independent spectrum-scoped entry point. It does not require a
time/current channel or serialize spectra as BDF time-series artifacts.

## Installation and entry points

Install the EIS providers for Battery Feature Lab 0.7.0:

```console
python -m pip install "battery-feature-lab[eis]==0.7.0"
```

For development from a source checkout:

```console
python -m pip install -e ".[eis]"
```

The extra pins `eisfit==0.2.0`, `impedance==1.7.1` and NumPy 1.26.x.
`impedance 1.7.1` builds an evaluated circuit expression that is incompatible
with NumPy 2 scalar representations. The extra also keeps pyarrow below 26:
pyarrow 26 requires NumPy 2 without declaring it, so pip would otherwise
install a pair that fails at import. The compatible versions are locked;
provider failures do not trigger a replacement implementation.

```python
import bfl

result = bfl.analyze_eis(
    "spectra.csv",
    output_dir="eis_out",
    group_columns=("Cell_ID", "Spectrum_ID"),
    metadata_path="measurement_conditions.json",
    fit_options={"tail": "cpe", "n_starts": 20, "seed": 7},
)
```

For a file known to contain exactly one sweep:

```console
bfl analyze-eis spectrum.csv --single-spectrum --output-dir eis_out
```

The CLI accepts `--sheet`, repeated `--group-column`, `--phase-unit`,
`--metadata-path`, `--no-kk`, `--no-fit`, `--tail`, `--n-starts` and `--max-nfev`.
The Python interface additionally exposes exact column mappings, probe
frequencies, provider configuration and comparison pairs.

## Input identity, units and provenance

BDS reads the selected table and normalizes the canonical impedance columns.
The EIS adapter identifies spectra before normalization, carries source-row
identifiers through sorting, and records every excluded point and its reason.
Row numbers are zero-based positions in the BDS-parsed table, not physical
Excel row numbers. Sheet selection and the source file digest are retained.

Recorded `Spectrum_ID`, `Measurement_ID` or `Sweep_ID` is used together with
`Cell_ID` where available. Otherwise, supply grouping columns or explicitly
declare a single spectrum. A single-spectrum declaration cannot override
multiple recorded sweeps. Incomplete grouping keys and varying per-spectrum
condition fields fail rather than silently combining measurements.

Column mappings declare frequency in Hz and impedance in Ohm. Known
non-canonical unit headers are refused. Magnitude/phase data require a degree
or radian declaration in the header or `phase_unit`; phase values are never
used to guess the unit. Negative-imaginary headers are converted once and the
sign operation is recorded. No interpolation, averaging or fabricated cycle
identity is used during intake. Advanced source layouts should first be
exported to a supported table with explicit quantities and units.

## Declared conditions

```json
{
  "defaults": {
    "cell_id": "cell_001",
    "temperature_deg_c": 25,
    "soc_fraction": 0.5,
    "rest_time_s": 3600,
    "ac_amplitude_v": 0.01,
    "instrument": "instrument_001",
    "protocol_id": "eis_protocol_001"
  },
  "spectra": {
    "spectrum_02": {"cycle_index": 100}
  }
}
```

Declared and source-column values must agree where both are present. Missing
conditions remain `not_reported`. Source condition columns represent constant
per-sweep metadata. `SOC_fraction` is a fraction, not a percentage. Voltage and
current perturbation amplitudes remain distinct quantities. Metadata files
are identified by portable filename and SHA-256; inline declarations are
retained in the run configuration. Spectrum IDs must be unambiguous across
cells for metadata and comparisons, or callers must use `spectrum_uid`.

## Operation, Response and Evolution

- Operation records the declared/reported conditions and their sources.
- Response retains the measured spectrum, source-row references, numerical QC,
  frequency coverage and probe values. Probe interpolation is linear in real
  and imaginary impedance against log10 frequency, with no extrapolation.
- Lin-KK records provider version, settings, normalized residuals and the
  configured residual-limit outcome. This is a consistency test, not proof of
  linearity, stationarity or a particular circuit interpretation. By default
  Lin-KK includes a series capacitance (`add_cap=True`), because battery spectra
  usually end in a capacitive or diffusion tail with no low-frequency
  real-axis intercept; without it the lowest frequencies are flagged. Declare
  `kk_options={"add_cap": False}` for spectra that do reach that intercept. Metrics
  locate the result: `frequency_of_maximum_residual` (Hz),
  `points_above_residual_limit` and `fraction_points_above_residual_limit`.
  A largest residual at one frequency does not show that the remaining points
  pass a re-run without that point; Lin-KK places its time constants from the
  measured frequency range.
- When `impedance.py` returns a non-finite μ or non-finite or misshaped arrays,
  the Lin-KK record is `not_computable` with reason
  `provider_invalid_result:lin_kk_mu_not_finite` or
  `provider_invalid_result:lin_kk_invalid_outputs`. The record keeps the
  observed facts under `attributes.lin_kk_termination`: the RC element count at
  termination, μ when finite, its class (`nan`, `positive_infinity` or
  `negative_infinity`) and the invalid outputs. `impedance.py` computes
  μ = 1 − Σ|negative RC resistances| / Σ|non-negative RC resistances|, and its
  automatic element-count search also ends when μ is non-finite. BFL reports
  this outcome; it does not substitute another Lin-KK implementation or infer a
  measurement cause.
- ECM records the circuit, parameter units, effective bounds, random seed,
  weighting, loss, multistart audit, convergence, fit error and residuals.
  Failed fits retain the measured spectrum and produce an explicit unavailable
  ECM record. Duplicate grids and insufficient point counts skip providers.
- The multistart audit keeps each start's start and final parameters (final
  values in the reported arc order). `attributes.multistart_near_best`, also in
  the compact index, gives the starts whose cost is within the eisfit
  `near_best_rtol` (fit option, default 0.01) of the best, and the range of each
  parameter across them. These are recorded facts about the optimizer run;
  BFL applies no stability threshold to them. When `start_count` is 1 the ranges
  are zero by construction and say nothing about stability; it only shows that a
  single start reached the best cost within the tolerance.
- Parameter stability, uncertainty and identifiability remain `not_assessed`.
  A converged fit, a canonical arc label or agreement among near-best starts
  does not establish those properties.

Residual limits and the default minimum of 12 provider points are configurable
analysis policies, not universal scientific acceptance criteria. A failing or
unavailable consistency check flags the fit's measurement evidence separately
from its convergence and fit error.

Comparisons are explicit:

```python
result = bfl.analyze_eis(
    "spectra.csv", output_dir="comparison_out",
    metadata_path="measurement_conditions.json",
    comparison_pairs=(("spectrum_01", "spectrum_02"),),
    comparison_kind="ageing",
    comparison_tolerances={"temperature_deg_c": 0.5},
)
```

`repeat` and `ageing` require matched recorded conditions. `ageing` additionally
requires increasing recorded cycle, elapsed-time or sequence order. `soc`,
`temperature` and `specimen` permit only the named axis (SOC, temperature or
`cell_id`) to differ; remaining conditions must match. `specimen` compares two
cells measured under the same conditions; its differences combine cell-to-cell
variation and any unrecorded history and are not attributed to one cause. Other
kinds require the same `cell_id`. Unknown conditions produce `conditional`;
conflicting conditions or ambiguous frequency overlap produce `incompatible`.
Observed impedance differences are emitted only after the condition gate passes,
within both spectra's frequency coverage.

Numeric conditions match within a tolerance. Measured temperature and SOC
default to 0.5 °C and 0.01; protocol settings such as rest time, excitation
amplitude, DC levels and indices default to exact equality. Declared
`comparison_tolerances` override the defaults, and each comparison records the
effective tolerances and whether each was a default or declared.

`dc_current_a` (source column `DC_Current_A`) records the DC current during the
measurement, so steady-state and operando spectra are not compared as the same
measurement. When either spectrum records it, it is gated like the other
conditions. When neither does, it is listed in `unassessed_conditions` and does
not change the comparison status.

ECM parameter differences also require two applicable fits and remain
conditional on unassessed parameter stability and physical arc identity.
CPE-coefficient differences are withheld when their exponents, and therefore
their dimensional units, differ. No comparison is promoted to an ageing cause
or degradation mechanism.

## Output contract

Each run writes six artifacts:

```text
normalized_eis.parquet
eis_conversion_report.json
analysis_metadata.json
analysis_results.json
analysis_evidence.json
analysis_validation.json
```

The JSON contracts use modality-specific schemas: `bfl.eis-intake/0.1`,
`bfl.eis-analysis/0.1`, `bfl.eis-metadata/0.1`, `bfl.eis-validation/0.1` and
`bfl.eis-summary/0.2`. The compact results index uses the Operation/Response/
Evolution layout, while methods, dense spectra, fit residuals and detailed
diagnostics remain in evidence. Resolve `record_id` within its `run_id`; IDs
are run-scoped.

The compact index states repeated content once:

- `conditions.shared` holds condition blocks identical for every spectrum, and
  `conditions.not_reported` names conditions reported for no spectrum. Each
  `operation.eis_conditions` record lists only the conditions that differ
  between spectra.
- Comparison records name `compared_spectra` (reference first) instead of
  repeating both condition blocks.
- `interpretation_limits` maps identifiers such as `L1` to their text; each
  compact record lists identifiers.

Nothing is dropped: the evidence record with the same `record_id` keeps the
complete condition block and limit text.

Each `provider_calls` entry for an invoked Lin-KK or circuit-fit provider
includes `provider_output`: stdout and stderr line counts and the warning
categories, messages and counts emitted during the call. Calls that were not
invoked, or whose provider could not be imported, have no `provider_output`. Console text and warning source locations are not retained,
and absolute paths inside warning messages are replaced with `<path>`.
Validation checks finite JSON, schemas, reference identity, artifact digests
and complete retained/excluded row accounting. Serialized provenance contains
no host-local paths. Output directories must be empty.

These files are the handoff for downstream tools. BFL does not implement DRT
analysis.

The validation file records the installed provider versions and a digest of
the EIS implementation sources, so development checkouts remain distinguishable
even before the next BFL version is released.

## Upgrading eisfit

BFL pins an exact eisfit release in the `eis` extra and in `uv.lock`. A new eisfit
release is not picked up automatically; adopting it is an explicit change to BFL.

What follows the installed release without code changes: the provider version in
ECM records and validation, the `run_id`, the release reference (built from the
installed version, so the matching `v<version>` tag must exist on GitHub), the
parameter units, which eisfit provides, and any new content inside
`fit.diagnostics`, which is stored as returned.

What BFL still decides explicitly:

- the validated fit options (`ECM_FIT_OPTIONS`) and tail models (`ECM_TAILS`);
  new options need review before they can be passed through;
- that every fitted parameter has a unit: a parameter for which eisfit supplies
  none makes the ECM record `not_computable` with reason
  `provider_invalid_result:ecm_parameter_units_undeclared`, rather than being
  reported as dimensionless;
- the two-arc ECM method name, circuit description and parameter-comparison rules.
  A new circuit or tail model needs explicit adaptation, not only a version change.

To adopt a release: change the pin, run `uv lock --upgrade-package eisfit`, run the
test suite, including `tests/test_eisfit_contract.py`, which fits a small spectrum
for each tail and checks fields, shapes, units and JSON serialisation, then update
this page and the changelog. Most failures while reading provider results are
recorded as provider errors, but an incompatible release can still fail during
serialisation, which is why the contract test runs a real fit.

An exact pin supports reproducibility but does not guarantee it: `pip install`
does not read `uv.lock`, and other dependencies, configuration and platform also
affect results. Users of the published package receive a new eisfit only through
a new BFL release that declares it.

## Worked example

[`examples/BFL_Zhang_2020_EIS_example.ipynb`](../examples/BFL_Zhang_2020_EIS_example.ipynb) runs this
path on a public CC BY 4.0 dataset. It screens 261 spectra from one ageing coin cell, then fits and
compares six of them under the `ageing` gate.
