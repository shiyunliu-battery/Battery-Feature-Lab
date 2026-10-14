# Battery Feature Lab documentation

The documentation is intentionally small and follows the public product boundary.

- [Core architecture](architecture.md) explains preprocessing, capability gating, the three
  analytical dimensions, and the evidence pipeline.
- [API reference](api.md) documents `bfl.analyze()`, `bfl analyze`, the input adapters, the six output
  artifacts, and the stable result object.
- [EIS analysis and evidence](eis.md) documents `bfl.analyze_eis()`, spectrum identity, declared
  conditions, Lin-KK and circuit-fit records, comparison gates, and the EIS output contract.

For runnable end-to-end examples, open
[`examples/BFL_Catenaro_Onori_example.ipynb`](../examples/BFL_Catenaro_Onori_example.ipynb) (time
series) or [`examples/BFL_Zhang_2020_EIS_example.ipynb`](../examples/BFL_Zhang_2020_EIS_example.ipynb)
(EIS). [`examples/README.md`](../examples/README.md) indexes both.

## Stable interfaces

| Interface | Version |
|---|---|
| Package | `0.6.0` |
| Detailed evidence | `bfl.analysis/0.1` |
| Compact summary | `bfl.summary/0.1` |
| Metadata | `bfl.metadata/0.1` |
| Validation | `bfl.validation/0.1` |
| Analysis policy | `bfl.analysis-policy/0.1` |
| EIS intake report | `bfl.eis-intake/0.1` |
| EIS detailed evidence | `bfl.eis-analysis/0.1` |
| EIS compact summary | `bfl.eis-summary/0.2` |
| EIS metadata | `bfl.eis-metadata/0.1` |
| EIS validation | `bfl.eis-validation/0.1` |

Provider versions in evidence records are read from the installed package metadata, for example
`BFL 0.6.0` or `PyProBE <version>; BFL 0.6.0`.

BFL emits machine-readable JSON and Parquet. Natural-language rendering belongs to downstream
software.
