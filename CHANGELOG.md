# Changelog

## 0.3.0

### Ingest is now delegated to BDS

BFL no longer parses cycler files. `battery-data-standard` (BDS) is a required dependency
and owns file reading, cycler detection, column mapping, unit conversion, time-axis repair,
current-sign resolution and schema validation.

**Removed**

- `battery_feature_lab/core/units.py`. BDS performs unit normalization.
- The hand-rolled table parser, alias matcher and unit logic in `bds_adapter/readers.py`.

**Changed**

- `read_bds_export()` keeps its signature and return contract. It now reads through
  `bds.read_with_report()`.
- Input coverage now equals whatever the installed BDS version supports. Run `bds formats`
  to see the current list. JSON and JSONL are staged to CSV and handed to BDS.
- `run_metadata.json` gains a `bds_conversion_report` field carrying adapter identity,
  detection confidence, evidence tier, unit transforms, repairs and warnings.
- Files BDS cannot read now raise a `ValueError` naming `bds doctor` rather than failing
  later with a missing-column error.
- `bds_adapter/validators.py` now checks only the BFL-specific fields (`cell_id`,
  `cycle_index`, `step_type`). Time, voltage and current are validated by BDS.

**Behaviour change**

- `ReaderConfig.positive_current_is_charge` is ignored and raises a `DeprecationWarning`.
  BDS resolves the source current-sign convention from the adapter and normalizes to
  charge-positive. Setting this flag previously flipped the sign after parsing, which would
  now double-correct data that BDS has already normalized.
- `ReaderConfig.time_unit` and `ReaderConfig.capacity_unit` no longer take effect. BDS
  derives units from the source columns. `soc_unit` is still honoured because `soc` is not
  part of the BDS schema.

**Added**

- `read_bds_export_with_report()` returns the frame and the BDS conversion report.
- `power_w` and `step_time_s` now appear in the normalized table when BDS can derive them.

### Migration

No code change is required for the documented API. If you set `positive_current_is_charge`,
`time_unit` or `capacity_unit`, remove them and verify one representative file with
`bds explain <file> --text` before running a batch.
