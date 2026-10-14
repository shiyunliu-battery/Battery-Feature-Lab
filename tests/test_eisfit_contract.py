"""Contract between BFL and the pinned eisfit release.

An eisfit upgrade must pass these tests before the exact pin in pyproject.toml changes.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import tomllib
import typing
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from battery_feature_lab.eis.analysis import eisfit_reference
from battery_feature_lab.eis.schema import ECM_FIT_OPTIONS, ECM_TAILS

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("eisfit") is None,
    reason="eisfit contract tests require battery-feature-lab[eis]",
)

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
TRUE_PARAMETERS = {
    "cpe": [1e-7, 0.014, 0.003, 0.7, 0.82, 0.0015, 1.2, 0.9, 700.0, 0.55],
    "warburg": [1e-7, 0.014, 0.003, 0.7, 0.82, 0.0015, 1.2, 0.9, 0.01],
}


def pinned_version(package):
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    matches = [
        re.fullmatch(rf"{re.escape(package)}==(\S+)", requirement)
        for requirement in project["optional-dependencies"]["eis"]
    ]
    return next(match.group(1) for match in matches if match)


def test_installed_eisfit_matches_the_exact_pin_and_release_reference():
    pinned = pinned_version("eisfit")
    assert version("eisfit") == pinned
    assert eisfit_reference().endswith(f"/releases/tag/v{pinned}")


def test_validated_fit_options_and_tails_exist_in_eisfit():
    import eisfit

    assert ECM_FIT_OPTIONS <= {field.name for field in dataclasses.fields(eisfit.FitConfig)}
    assert set(typing.get_args(eisfit.TailModel)) == ECM_TAILS


@pytest.mark.parametrize("tail", sorted(ECM_TAILS))
def test_fit_result_provides_every_field_bfl_reads(tail):
    import eisfit

    names = eisfit.parameter_names(tail)
    frequency = np.logspace(4, -1, 32)
    impedance = eisfit.model_eis(np.array(TRUE_PARAMETERS[tail]), 2 * np.pi * frequency, tail)
    options = eisfit.FitConfig(tail=tail, n_starts=2, seed=7)
    fit = eisfit.fit_model_eis(frequency, impedance, options)

    assert fit.optimizer.success in (True, False)
    assert np.isfinite(fit.nrmse) and np.isfinite(fit.rmse)
    assert fit.Zsim.shape == impedance.shape
    assert list(fit.parameters) == names
    assert all(np.isfinite(value) for value in fit.parameters.values())
    assert list(fit.parameter_units) == names
    assert all(isinstance(unit, str) and unit for unit in fit.parameter_units.values())
    assert hasattr(fit, "bound_hits")
    near_best = fit.diagnostics["near_best"]
    assert {"relative_cost_tolerance", "start_count", "start_indices", "parameter_ranges"} <= set(
        near_best
    )
    assert near_best["start_count"] >= 1 and set(near_best["parameter_ranges"]) == set(names)
    json.dumps(fit.diagnostics, allow_nan=False)
    json.dumps([dataclasses.asdict(attempt) for attempt in fit.attempts], allow_nan=False)
    json.dumps(dataclasses.asdict(options), allow_nan=False)
