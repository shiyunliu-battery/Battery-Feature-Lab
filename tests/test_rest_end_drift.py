"""End-of-rest drift, preceding-load end voltage and uncounted wall-clock time in rest records."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import bfl
from battery_feature_lab.analysis import response
from battery_feature_lab.analysis.schema import ANALYSIS_POLICY_VERSION, AnalysisConfig


def _rest_frames(*, wall: bool = True, pause_at: int | None = None, pause_s: float = 2700.0,
                 rest_samples: int = 241, dt: float = 30.0, duplicate_last: bool = False):
    """A 20-sample discharge followed by a rest whose voltage rises 12 mV/h at the end."""
    load = 20
    n = load + rest_samples
    time = np.arange(n, dtype=float) * dt
    current = np.r_[np.full(load, -2.5), np.zeros(rest_samples)]
    rest_time = time[load:] - time[load]
    rest_voltage = 2.9 - 0.3 * np.exp(-rest_time / 600.0) + 0.012 * rest_time / 3600.0
    voltage = np.r_[np.linspace(2.8, 2.5, load), rest_voltage]
    if duplicate_last:
        time[-1] = time[-2]
        voltage[-2] = 9.0  # the earlier of two samples at one time stamp must not be used
    columns = {"_source_row": np.arange(n), "test_time_s": time, "current_a": current, "voltage_v": voltage}
    if wall:
        clock = 1_549_465_080.0 + time
        if pause_at is not None:
            clock[load + pause_at:] += pause_s
        columns["Unix Time / s"] = np.floor(clock / 60.0) * 60.0
    frame = pl.DataFrame(columns)
    py_frame = pl.DataFrame({
        "_source_row": np.arange(n), "Time [s]": time, "Current [A]": current, "Voltage [V]": voltage,
        "Cycle": np.ones(n, dtype=int), "Step": np.r_[np.zeros(load, dtype=int), np.ones(rest_samples, dtype=int)],
        "Event": np.r_[np.zeros(load, dtype=int), np.ones(rest_samples, dtype=int)],
        "Capacity [Ah]": np.linspace(0.0, 0.1, n),
    })
    phases = [
        {"segment_index": 0, "start": 0, "end": load, "phase": "discharge", "duration_s": load * dt,
         "mode": "constant_current_like", "source_interval": {"start_row": 0, "end_row": load - 1}},
        {"segment_index": 1, "start": load, "end": n, "phase": "rest", "previous_phase": "discharge",
         "duration_s": rest_samples * dt, "source_interval": {"start_row": load, "end_row": n - 1}},
    ]
    return frame, py_frame, phases


def _rest_record(monkeypatch, frames, config=None):
    frame, py_frame, phases = frames
    load = phases[1]["start"]

    class RestExperiment:
        def rest(self):
            return SimpleNamespace(data=py_frame.slice(load, py_frame.height - load))

    monkeypatch.setattr(response, "pyprobe_experiment", lambda _frame: RestExperiment())
    records = response.analyze_rest_and_thermal(
        frame, py_frame, phases, config=config or AnalysisConfig(), cell_id="cell", cycle_id=None,
        provider_calls=[],
    )
    assert len(records) == 1
    return records[0]


def test_end_drift_is_the_slope_over_the_final_window(monkeypatch) -> None:
    record = _rest_record(monkeypatch, _rest_frames())
    metrics = record["metrics"]
    # over the last 600 s the exponential term still adds about 2e-5 V/h to the 12 mV/h ramp
    # (0.3 V / 600 s * exp(-11..-12) * 3600 s/h)
    assert metrics["end_drift_rate"]["unit"] == "V/h"
    assert metrics["end_drift_rate"]["value"] == pytest.approx(0.012, abs=3e-5)
    assert metrics["end_drift_rate"]["value"] > 0.012
    assert metrics["end_window_sample_count"]["value"] == 21
    assert metrics["end_window_span"]["value"] == pytest.approx(600.0)
    assert metrics["preceding_load_end_voltage"]["value"] == pytest.approx(2.5)
    assert record["method"]["name"] == "rest_filter_with_interval_descriptors_v2"
    parameters = record["method"]["parameters"]
    assert parameters["end_window_s"] == 600.0 and parameters["end_window_min_samples"] == 3
    assert parameters["analysis_policy_version"] == ANALYSIS_POLICY_VERSION == "bfl.analysis-policy/0.3"


def test_uncounted_wall_time_inside_the_rest_is_measured_and_flagged(monkeypatch) -> None:
    record = _rest_record(monkeypatch, _rest_frames(pause_at=100))
    metrics = record["metrics"]
    assert metrics["uncounted_wall_time"]["value"] == pytest.approx(2700.0, abs=60.0)
    assert metrics["uncounted_wall_time_interval_count"]["value"] == 1
    assert "test_clock_pause_within_interval" in record["quality"]["flags"]
    assert record["attributes"]["wall_clock_column"] == "Unix Time / s"
    assert record["method"]["parameters"]["wall_clock_resolution_s"] == 60.0
    assert record["method"]["parameters"]["clock_divergence_threshold_s"] == 120.0
    # the test-clock duration does not include the pause
    assert metrics["duration"]["value"] == pytest.approx(241 * 30.0)


def test_minute_rounding_of_the_wall_clock_is_not_a_pause(monkeypatch) -> None:
    record = _rest_record(monkeypatch, _rest_frames())
    assert record["metrics"]["uncounted_wall_time"]["value"] == 0
    assert "test_clock_pause_within_interval" not in record["quality"]["flags"]


def test_without_a_wall_clock_uncounted_time_is_not_computable(monkeypatch) -> None:
    record = _rest_record(monkeypatch, _rest_frames(wall=False))
    metric = record["metrics"]["uncounted_wall_time"]
    assert metric["status"] == "not_computable" and metric["reason"] == "no_wall_clock_channel"
    assert record["attributes"]["wall_clock_column"] is None


def test_short_rest_has_no_end_drift(monkeypatch) -> None:
    record = _rest_record(monkeypatch, _rest_frames(rest_samples=2))
    metric = record["metrics"]["end_drift_rate"]
    assert metric["status"] == "not_computable"
    assert metric["reason"] == "fewer_than_minimum_samples_in_end_window"


def test_the_last_sample_of_a_duplicated_time_stamp_is_used(monkeypatch) -> None:
    record = _rest_record(monkeypatch, _rest_frames(duplicate_last=True))
    assert record["metrics"]["end_drift_rate"]["value"] == pytest.approx(0.012, rel=0.05)


def test_window_and_threshold_are_policy_values(monkeypatch) -> None:
    config = AnalysisConfig(analysis_policy={"relaxation_end_window_s": 300.0, "clock_divergence_min_s": 5000.0})
    record = _rest_record(monkeypatch, _rest_frames(pause_at=100), config=config)
    assert record["metrics"]["end_window_sample_count"]["value"] == 11
    assert record["metrics"]["uncounted_wall_time"]["value"] == 0
    with pytest.raises(ValueError):
        AnalysisConfig(analysis_policy={"relaxation_end_window_min_samples": 2.5})


def test_full_run_writes_the_new_metrics(vendor_csv, tmp_path) -> None:
    result = bfl.analyze(vendor_csv, output_dir=tmp_path / "out", input_adapter="bds")
    import json

    evidence = json.loads(result.analysis_evidence_path.read_text(encoding="utf-8"))
    rests = [r for r in evidence["records"] if r["record_type"] == "response.rest_and_thermal"]
    assert rests
    for record in rests:
        metrics = record["metrics"]
        assert {"end_drift_rate", "preceding_load_end_voltage", "uncounted_wall_time"} <= set(metrics)
        assert metrics["uncounted_wall_time"]["status"] == "not_computable"
