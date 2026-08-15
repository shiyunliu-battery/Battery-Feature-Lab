# Local real-data validation

Raw datasets in this directory are local-only and ignored by Git. Keep only this README and `MANIFEST.sha256` under version control.

Run:

```console
python scripts/validate_real_data.py path/to/source.xlsx --output-root .real-validation --temperature-column "raw:Surface_Temp(degC)"
```

The validator uses `bfl.analyze()` directly. Raw cycler files use the BDS
adapter; native formal BDF artifacts can use `--input-adapter bdf` when the
`battery-feature-lab[bdf]` extra is installed. Provider failures do not trigger cross-adapter
fallback. An unsupported raw format must be resolved with a documented BDS
adapter or a correctly documented source export.

## Current local validation status

`NCA_k1_0_05C_05degC.xlsx` was validated locally with BDS 0.3.1 and the explicit
`raw:Surface_Temp(degC)` temperature channel. The run preserved all 93,305 rows, produced the six
artifacts, passed every JSON Schema check, resolved every compact evidence reference, and reproduced
the selected capacity and relaxation identities with zero absolute error. The source workbook remains
Git-ignored; its digest is recorded in `MANIFEST.sha256`.

The two MATLAB files still stop at BDS ingest. They expose only `ExpIR` and `Expivium`; BDS cannot map
those unnamed arrays to required time, voltage and current semantics. No column order is assumed by
BFL. Supporting those files requires either:

1. the source is re-exported with named quantities and units supported by BDS, or
2. an explicit BDS adapter with documented `Expivium` column semantics and a reduced, publishable
   regression fixture.
