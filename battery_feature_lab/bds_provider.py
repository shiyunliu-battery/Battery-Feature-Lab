"""Resolve BDS distribution identity without changing its compatible import API."""

from importlib.metadata import PackageNotFoundError, version


def bds_distribution() -> tuple[str, str]:
    """Require exactly one owner of the shared bds/battery_data_standard modules."""
    installed = []
    for name in ("battery-data-steward", "battery-data-standard"):
        try:
            installed.append((name, version(name)))
        except PackageNotFoundError:
            pass
    if len(installed) > 1:
        raise RuntimeError(
            "Both battery-data-steward and battery-data-standard are installed and share "
            "Python modules. Create a clean environment with exactly one BDS distribution; "
            "uninstalling only one can remove files needed by the other."
        )
    if not installed:
        raise PackageNotFoundError("Install one supported BDS distribution")
    return installed[0]
