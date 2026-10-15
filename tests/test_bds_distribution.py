"""Distribution rename must preserve accurate provenance and reject shared-file installs."""

import pytest

from battery_feature_lab import bds_provider


@pytest.mark.parametrize("name", ["battery-data-standard", "battery-data-steward"])
def test_single_distribution_identity(monkeypatch, name):
    def installed(package):
        if package == name:
            return "test-version"
        raise bds_provider.PackageNotFoundError(package)

    monkeypatch.setattr(bds_provider, "version", installed)
    assert bds_provider.bds_distribution() == (name, "test-version")


def test_overlapping_distributions_fail_with_recovery_instructions(monkeypatch):
    monkeypatch.setattr(bds_provider, "version", lambda package: "test-version")
    with pytest.raises(RuntimeError, match="clean environment"):
        bds_provider.bds_distribution()


def test_missing_distribution_is_not_misattributed(monkeypatch):
    def missing(package):
        raise bds_provider.PackageNotFoundError(package)

    monkeypatch.setattr(bds_provider, "version", missing)
    with pytest.raises(bds_provider.PackageNotFoundError):
        bds_provider.bds_distribution()
