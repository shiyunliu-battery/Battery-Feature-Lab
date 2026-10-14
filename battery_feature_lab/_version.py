"""Installed Battery Feature Lab version, the single source for provenance records."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("battery-feature-lab")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0+unknown"

PROVIDER_VERSION = f"BFL {__version__}"
