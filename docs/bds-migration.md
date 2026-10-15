# Battery Data Steward migration validation

Checked on 15 October 2026 against BFL main `d6433c4`.

## Publication and dependency decision

The PyPI simple index resolves `battery-data-steward==0.4.0`; its wheel was
uploaded at 2026-10-14T22:38:30.990Z. A clean Python 3.12 environment installed
that published wheel successfully. `uv.lock` records the public artifact URL,
SHA-256, and upload time. This is a released package, not just the renamed
repository source (`6802e688162aab18f3db635fa26dae407878b8ab`).
Therefore this revision switches the mandatory dependency to
`battery-data-steward>=0.4.0,<0.5` and updates the lock without unrelated upgrades.
BFL source is 0.7.0; the public BFL PyPI page checked during this work still
served 0.5.0. This PR neither publishes nor changes the BFL version.

## Isolated compatibility checks

Two separate Python 3.12 environments exercised the full test suite with EIS
extras. One contained only the published legacy BDS 0.3.1 distribution, the other
only the published Steward 0.4.0 distribution. Both passed the cycling and EIS
paths, conversion reports, source mapping, evidence hashes, and existing CLI
regressions. The added installed-provider integration test runs both modalities
and asserts the actual distribution name/version in generated artifacts.

The old-provider run is an API compatibility check with an intentionally replaced
dependency, not a claim that the new BFL dependency metadata resolves to legacy
BDS. The new-provider environment also passed `uv pip check`. CI runs both
provider variants on Python 3.11 and 3.12; local execution used Python 3.12.
Synthetic fixtures and the repository's included EIS fixture are covered; this
is not a new broad real-world vendor dataset qualification.

## Stable interfaces and provenance

- Keep `import bds` and `battery_data_standard.eis/io`: these remain supported
  module paths in the renamed distribution.
- Keep `--input-adapter bds`, report filenames, canonical channel names, and
  historical format identifiers such as `battery-data-standard-export-v1`.
- Resolve distribution metadata centrally. Cycling handoffs, metadata tool
  sources, EIS intake and both software-version manifests use the installed
  distribution name and version rather than assuming legacy metadata exists.
- Preserve recorded example outputs and their historical provider/version.
  Notebook executable version queries use the current installed identity.
- Reject coexistence before processing: both distributions own shared Python
  files. Metadata alone cannot identify which package last overwrote them.

## Upgrade and release procedure

Create a fresh environment and install the BFL revision/release containing this
change. Do not add Steward alongside an installed legacy BFL release: pip can
retain both distributions because their project names differ. If both were
installed, recreate the environment; removing only one can damage the other.

Before publishing the next BFL release, require the provider-matrix CI to pass
and install its built wheel in a clean environment. Confirm only Steward is
installed and `pip check` succeeds. A future BDS minor release outside `<0.5`
requires a separate compatibility decision. No data-format migration is needed.
