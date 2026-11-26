"""BFL uses Battery Data Steward and refuses the retired battery-data-standard distribution."""

import pytest

from battery_feature_lab import bds_provider


def _only(*names):
    def installed(package):
        if package in names:
            return "test-version"
        raise bds_provider.PackageNotFoundError(package)

    return installed


def test_steward_identity_is_reported(monkeypatch):
    monkeypatch.setattr(bds_provider, "version", _only("battery-data-steward"))
    assert bds_provider.bds_distribution() == ("battery-data-steward", "test-version")


def test_retired_distribution_alone_is_refused(monkeypatch):
    monkeypatch.setattr(bds_provider, "version", _only("battery-data-standard"))
    with pytest.raises(RuntimeError, match="former name of battery-data-steward"):
        bds_provider.bds_distribution()


def test_overlapping_distributions_fail_with_recovery_instructions(monkeypatch):
    monkeypatch.setattr(bds_provider, "version", lambda package: "test-version")
    with pytest.raises(RuntimeError, match="clean environment"):
        bds_provider.bds_distribution()


def test_missing_distribution_is_not_misattributed(monkeypatch):
    monkeypatch.setattr(bds_provider, "version", _only())
    with pytest.raises(bds_provider.PackageNotFoundError):
        bds_provider.bds_distribution()
