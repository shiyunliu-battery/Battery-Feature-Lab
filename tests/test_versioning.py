"""Provenance version strings come from installed package metadata."""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

import battery_feature_lab
from battery_feature_lab._version import PROVIDER_VERSION

RELEASE_LITERAL = re.compile(r"[\"'](?:BFL )?\d+\.\d+\.\d+[\"']|BFL \d+\.\d+\.\d+")


def test_package_version_matches_installed_metadata():
    assert battery_feature_lab.__version__ == version("battery-feature-lab")
    assert PROVIDER_VERSION == f"BFL {version('battery-feature-lab')}"


def test_package_sources_do_not_hard_code_release_versions():
    root = Path(battery_feature_lab.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in sorted(root.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if RELEASE_LITERAL.search(line)
    ]
    assert offenders == []
