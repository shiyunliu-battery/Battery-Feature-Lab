# EIS analysis

EIS has an independent spectrum-scoped entry point. It does not require a
time/current channel or serialize spectra as BDF time-series artifacts.

## Installation and entry points

This interface is part of `0.5.0.dev0`, which is currently unreleased. From a
checkout containing this change:

```console
python -m pip install -e ".[eis]"
```

After the next package release:

```console
python -m pip install "battery-feature-lab[eis]>=0.5,<0.6"
```

The extra pins `eisfit==0.1.1`, `impedance==1.7.1` and NumPy 1.26.x.
`impedance 1.7.1` builds an evaluated circuit expression that is incompatible
with NumPy 2 scalar representations. The compatible versions are locked;
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
  linearity, stationarity or a particular circuit interpretation.
- ECM records the circuit, parameter units, effective bounds, random seed,
  weighting, loss, multistart audit, convergence, fit error and residuals.
  Failed fits retain the measured spectrum and produce an explicit unavailable
  ECM record. Duplicate grids and insufficient point counts skip providers.
- Parameter stability, uncertainty and identifiability remain `not_assessed`.
  A converged fit or a canonical arc label does not establish those properties.

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
requires increasing recorded cycle, elapsed-time or sequence order. `soc` or
`temperature` permit only the named axis to differ; remaining conditions must
match. Unknown conditions produce `conditional`; conflicting conditions or
ambiguous frequency overlap produce `incompatible`. Observed impedance
differences are emitted only after the condition gate passes, within both
spectra's frequency coverage. Cross-cell comparisons are not admitted by this
initial same-cell policy.

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

The JSON contracts use the modality-specific `bfl.eis-* /0.1` schemas. The
compact results index uses the existing Operation/Response/Evolution layout,
while methods, dense spectra, fit residuals and detailed diagnostics remain
in evidence. Resolve `record_id` within its `run_id`; IDs are run-scoped.
Validation checks finite JSON, schemas, reference identity, artifact digests
and complete retained/excluded row accounting. Serialized provenance contains
no host-local paths. Output directories must be empty.

These files are the handoff for a downstream Battery Context adapter. This
change does not implement the Battery Context consumer or DRT analysis.

The validation file records the installed provider versions and a digest of
the EIS implementation sources, so development checkouts remain distinguishable
even before the next BFL version is released.
