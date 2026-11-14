"""Per-segment observations in operation.phase_segment records.

Each phase segment carries terminal values and ranges of voltage and temperature, absolute
current statistics, capacity and energy by previous-sample ZOH (over the segment's support) and
by the trapezoid rule (between its first and last samples), the source's accumulator at its first
and last samples, and the trapezoid integral minus the accumulator's change. These tests use a
synthetic CC discharge, rest, CC charge and CV charge whose accumulator continues from the CC
step into the CV step, as Maccor files do.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import bfl
from battery_feature_lab.analysis import operation
from battery_feature_lab.analysis.schema import AnalysisConfig

DT = 10.0


def _frame(*, reported: bool = True, directional: bool = False, reset_inside_cv: bool = False,
           ambient: bool = True) -> pl.DataFrame:
    """Discharge 2.5 A for 600 s, rest 300 s, CC charge 1.5 A for 600 s, CV at 4.2 V for 300 s."""

    blocks = [  # (step, samples, current at each sample, voltage at each sample)
        (1, 61, lambda k: -2.5, lambda k: 3.9 - 0.9 * k / 60),
        (2, 31, lambda k: 0.0, lambda k: 3.1 + 0.1 * (1 - np.exp(-k / 5))),
        (3, 61, lambda k: 1.5, lambda k: 3.3 + 0.9 * k / 60),
        (4, 31, lambda k: 1.5 * np.exp(-k / 10), lambda k: 4.2),
    ]
    rows: dict[str, list] = {name: [] for name in (
        "test_time_s", "current_a", "voltage_v", "step_index", "reported_q", "reported_e",
        "temperature_t1_deg_c", "ambient_temperature_deg_c")}
    t = 0.0
    q = e = 0.0
    previous_step = None
    for step, samples, current, voltage in blocks:
        if step in (1, 3):  # the source resets at a change of direction, not at the CV step
            q = e = 0.0
        for k in range(samples):
            if rows["test_time_s"] and previous_step == step:
                i_prev, v_prev = abs(rows["current_a"][-1]), rows["voltage_v"][-1]
                i_now, v_now = abs(current(k)), voltage(k)
                q += 0.5 * (i_prev + i_now) * DT / 3600.0
                e += 0.5 * (i_prev * v_prev + i_now * v_now) * DT / 3600.0
            if reset_inside_cv and step == 4 and k == 15:
                q = e = 0.0
            rows["test_time_s"].append(t)
            rows["current_a"].append(float(current(k)))
            rows["voltage_v"].append(float(voltage(k)))
            rows["step_index"].append(step)
            rows["reported_q"].append(q)
            rows["reported_e"].append(e)
            rows["temperature_t1_deg_c"].append(25.0 + (3.0 * k / samples if step == 1 else 0.5))
            rows["ambient_temperature_deg_c"].append(24.0)
            t += DT
            previous_step = step
    n = len(rows["test_time_s"])
    frame = pl.DataFrame(rows).with_columns(
        pl.Series("_source_row", np.arange(n)), pl.Series("record_index", np.arange(1, n + 1))
    )
    if directional:
        frame = frame.with_columns(
            pl.when(pl.col("current_a") > 0).then(pl.col("reported_q")).otherwise(None).alias("charge_capacity_ah"),
            pl.col("reported_q").alias("source_capacity_ah"),
        )
    elif reported:
        frame = frame.rename({"reported_q": "source_capacity_ah", "reported_e": "source_energy_wh"})
    if not ambient:
        frame = frame.drop("ambient_temperature_deg_c")
    return frame.drop([c for c in ("reported_q", "reported_e") if c in frame.columns])


def _segments(frame: pl.DataFrame) -> list[dict]:
    records, _ = operation.analyze_operation(
        frame, None, config=AnalysisConfig(), cell_id="cell", cycle_id=None, provider_calls=[]
    )
    return [r for r in records if r["record_type"] == "operation.phase_segment"]


def _value(record: dict, name: str):
    metric = record["metrics"][name]
    return metric["value"] if metric["status"] == "ok" else metric["reason"]


def test_terminal_values_ranges_and_current_statistics() -> None:
    discharge, rest, charge, cv = _segments(_frame())
    assert [s["attributes"]["phase"] for s in (discharge, rest, charge, cv)] == [
        "discharge", "rest", "charge", "charge"
    ]
    assert _value(discharge, "start_voltage") == pytest.approx(3.9)
    assert _value(discharge, "end_voltage") == pytest.approx(3.0)
    assert _value(discharge, "voltage_min") == pytest.approx(3.0)
    assert _value(discharge, "voltage_max") == pytest.approx(3.9)
    assert _value(discharge, "absolute_current_median") == pytest.approx(2.5)
    assert _value(cv, "absolute_current_max") == pytest.approx(1.5)
    assert _value(cv, "absolute_current_min") == pytest.approx(1.5 * np.exp(-3.0))
    assert _value(discharge, "start_temperature") == pytest.approx(25.0)
    assert _value(discharge, "temperature_max") == pytest.approx(25.0 + 3.0 * 60 / 61)
    assert _value(discharge, "ambient_temperature_mean") == pytest.approx(24.0)
    assert discharge["method"]["name"] == "source_step_and_current_phase_segmentation_v2"


def test_zoh_runs_to_the_next_segment_and_the_trapezoid_stops_at_the_last_sample() -> None:
    discharge = _segments(_frame())[0]
    # 61 samples 10 s apart: 600 s between the first and last sample, 610 s to the rest's first sample
    assert _value(discharge, "trapezoid_capacity") == pytest.approx(2.5 * 600 / 3600)
    assert _value(discharge, "zoh_capacity") == pytest.approx(2.5 * 610 / 3600)
    trapezoid_energy = 2.5 * 600 * (3.9 + 3.0) / 2 / 3600
    assert _value(discharge, "trapezoid_energy") == pytest.approx(trapezoid_energy)
    # a falling voltage held at the previous sample over-counts energy
    assert _value(discharge, "zoh_energy") > trapezoid_energy


def test_reconciliation_uses_the_change_of_the_accumulator_whatever_its_reset_scope() -> None:
    discharge, rest, charge, cv = _segments(_frame())
    assert _value(cv, "reported_capacity_start") == pytest.approx(_value(charge, "reported_capacity_end"))
    assert _value(cv, "reported_capacity_change") == pytest.approx(_value(cv, "trapezoid_capacity"))
    for segment in (discharge, rest, charge, cv):
        assert _value(segment, "trapezoid_capacity_minus_reported_change") == pytest.approx(0.0, abs=1e-12)
        assert _value(segment, "trapezoid_energy_minus_reported_change") == pytest.approx(0.0, abs=1e-12)
    assert discharge["attributes"]["observation_columns"]["reported_capacity"] == "source_capacity_ah"


def test_a_reset_inside_a_segment_is_not_reconciled() -> None:
    cv = _segments(_frame(reset_inside_cv=True))[3]
    assert _value(cv, "reported_capacity_change") == "reported_column_not_monotonic"
    assert _value(cv, "trapezoid_capacity_minus_reported_change") == "reported_column_not_monotonic"
    assert isinstance(_value(cv, "reported_capacity_end"), float)


def test_absent_channels_are_not_computable() -> None:
    discharge = _segments(_frame(reported=False, ambient=False))[0]
    for name in ("reported_capacity_start", "reported_energy_change", "trapezoid_capacity_minus_reported_change",
                 "start_ambient_temperature", "ambient_temperature_mean"):
        assert _value(discharge, name) == "channel_absent"
    assert discharge["attributes"]["observation_columns"]["ambient_temperature"] is None


def test_a_directional_accumulator_is_preferred_for_its_own_phase() -> None:
    discharge, _, charge, _ = _segments(_frame(directional=True))
    assert charge["attributes"]["observation_columns"]["reported_capacity"] == "charge_capacity_ah"
    assert discharge["attributes"]["observation_columns"]["reported_capacity"] == "source_capacity_ah"
    assert _value(charge, "trapezoid_capacity_minus_reported_change") == pytest.approx(0.0, abs=1e-12)


def test_full_run_writes_segment_observations(vendor_csv, tmp_path) -> None:
    result = bfl.analyze(vendor_csv, output_dir=tmp_path / "out", input_adapter="bds")
    evidence = json.loads(result.analysis_evidence_path.read_text(encoding="utf-8"))
    segments = [r for r in evidence["records"] if r["record_type"] == "operation.phase_segment"]
    assert segments
    names = {"start_voltage", "end_voltage", "absolute_current_median", "zoh_capacity", "trapezoid_capacity",
             "trapezoid_energy", "reported_capacity_change", "temperature_max"}
    for record in segments:
        assert names <= set(record["metrics"])
        assert record["metrics"]["trapezoid_capacity"]["unit"] == "Ah"
