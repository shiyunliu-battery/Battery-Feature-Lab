"""Resolve the Battery Data Steward distribution behind the ``bds`` import."""

from importlib.metadata import PackageNotFoundError, version

STEWARD = "battery-data-steward"
RETIRED = "battery-data-standard"


def bds_distribution() -> tuple[str, str]:
    """Return the installed Battery Data Steward name and version.

    ``battery-data-standard`` is the retired name of Battery Data Steward. Both
    distributions install the ``bds`` module, so an environment that still contains
    the retired one is refused even when Steward is also present.
    """
    try:
        retired = version(RETIRED)
    except PackageNotFoundError:
        retired = None
    if retired is not None:
        raise RuntimeError(
            f"{RETIRED} {retired} is installed. It is the former name of {STEWARD} and "
            "shares the 'bds' module with it. Create a clean environment with only "
            f"{STEWARD}>=0.7; uninstalling only one can remove files needed by the other."
        )
    try:
        return STEWARD, version(STEWARD)
    except PackageNotFoundError as exc:
        raise PackageNotFoundError(f"Install {STEWARD}>=0.7") from exc
