"""PyProBE constant-current/voltage candidate counts must be reproducible."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl

import bfl
from battery_feature_lab.analysis.operation import (
    PYPROBE_MODE_UNDEFINED,
    _mode_for_segment,
    _mode_policy_values,
    _pyprobe_modal_reference_defined,
)
from battery_feature_lab.analysis.schema import AnalysisConfig


def _discharge_frames(voltage: np.ndarray) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    count = len(voltage)
    time = np.arange(count, dtype=float)
    current = np.full(count, -1.0)
    frame = pl.DataFrame(
        {
            "_source_row": np.arange(count),
            "test_time_s": time,
            "current_a": current,
            "voltage_v": voltage,
            "cycle_index": np.ones(count, dtype=int),
        }
    )
    py_frame = pl.DataFrame(
        {
            "Time [s]": time,
            "Current [A]": current,
            "Voltage [V]": voltage,
            "Cycle": np.ones(count, dtype=int),
            "Step": np.ones(count, dtype=int),
            "Event": np.ones(count, dtype=int),
            "Capacity [Ah]": np.linspace(1.0, 0.9, count),
            "_source_row": np.arange(count),
        }
    )
    segment = {
        "segment_index": 0,
        "start": 0,
        "end": count,
        "phase": "discharge",
        "duration_s": float(count),
        "sample_count": count,
        "flags": [],
        "source_interval": {
            "start_row": 0,
            "end_row": count - 1,
            "start_time_s": 0.0,
            "end_time_s": float(count - 1),
        },
    }
    return frame, py_frame, segment


def _mode_policy() -> dict:
    return _mode_policy_values(AnalysisConfig())


def test_modal_reference_detects_all_distinct_values() -> None:
    distinct = pl.DataFrame({"Voltage [V]": np.linspace(4.1, 3.0, 200)})
    plateau = pl.DataFrame({"Voltage [V]": np.r_[np.full(50, 4.2), np.linspace(4.1, 3.0, 50)]})
    single = pl.DataFrame({"Voltage [V]": [3.7]})

    assert not _pyprobe_modal_reference_defined(distinct, "Voltage [V]")
    assert _pyprobe_modal_reference_defined(plateau, "Voltage [V]")
    assert _pyprobe_modal_reference_defined(single, "Voltage [V]")


def test_undefined_modal_reference_is_reported_not_counted() -> None:
    frame, py_frame, segment = _discharge_frames(np.linspace(4.1, 3.0, 200))
    outcomes = set()
    for _ in range(10):
        calls: list[dict] = []
        mode, attributes, flags = _mode_for_segment(
            frame, py_frame, segment, [segment], calls, _mode_policy()
        )
        outcomes.add((mode, repr(attributes), tuple(flags)))
        voltage_calls = [call for call in calls if call["method"] == "constant_voltage"]
        assert voltage_calls == [
            {
                "provider": "PyProBE",
                "method": "constant_voltage",
                "status": "not_invoked",
                "reason": PYPROBE_MODE_UNDEFINED,
                "source_interval": segment["source_interval"],
            }
        ]
        assert attributes["pyprobe_candidate_rows"]["constant_voltage"] is None
        assert attributes["pyprobe_candidate_rows"]["constant_current"] == 200
    assert len(outcomes) == 1
    assert next(iter(outcomes))[0] == "constant_current_like"


def test_repeated_analysis_writes_identical_evidence(vendor_csv: Path, tmp_path: Path) -> None:
    digests = set()
    for run in range(3):
        result = bfl.analyze(vendor_csv, tmp_path / f"run{run}", nominal_capacity_ah=1.0)
        digests.add(
            tuple(
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(result.files)
                if path.name in {"analysis_evidence.json", "analysis_validation.json"}
            )
        )
    assert len(digests) == 1
