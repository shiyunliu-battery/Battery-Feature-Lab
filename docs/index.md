# Battery Feature Lab documentation

The documentation is intentionally small and follows the public product boundary.

- [Core architecture](architecture.md) explains preprocessing, capability gating, the three
  analytical dimensions, and the evidence pipeline.
- [API reference](api.md) documents `bfl.analyze()`, `bfl analyze`, the input adapters, the six output
  artifacts, and the stable result object.

For a runnable end-to-end example, open
[`examples/BFL_Catenaro_Onori_example.ipynb`](../examples/BFL_Catenaro_Onori_example.ipynb).

## Stable interfaces

| Interface | Version |
|---|---|
| Package | `0.4.0` |
| Detailed evidence | `bfl.analysis/0.1` |
| Compact summary | `bfl.summary/0.1` |
| Metadata | `bfl.metadata/0.1` |
| Validation | `bfl.validation/0.1` |
| Analysis policy | `bfl.analysis-policy/0.1` |

BFL emits machine-readable JSON and Parquet. Natural-language rendering belongs to downstream
software.
