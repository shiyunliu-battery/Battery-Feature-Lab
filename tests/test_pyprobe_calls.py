"""Provider-call contract tests for pulse resistance and diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from battery_feature_lab.analysis import response
from battery_feature_lab.analysis.characterization import analyze_relaxation_summary
from battery_feature_lab.analysis.schema import AnalysisConfig


def _pulse_frames() -> tuple[pl.DataFrame, pl.DataFrame, list[dict]]:
    count = 82
    time = np.arange(count, dtype=float)
    current = np.r_[np.zeros(70), np.full(12, -1.0)]
    voltage = np.r_[np.full(70, 4.0), np.linspace(3.9, 3.85, 12)]
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
            "Step": np.r_[np.zeros(70, dtype=int), np.ones(12, dtype=int)],
            "Event": np.r_[np.zeros(70, dtype=int), np.ones(12, dtype=int)],
            "Capacity [Ah]": np.linspace(1.0, 0.99, count),
            "_source_row": np.arange(count),
        }
    )
    phases = [
        {
            "segment_index": 0,
            "start": 0,
            "end": 70,
            "phase": "rest",
            "duration_s": 70.0,
            "sample_count": 70,
            "source_interval": {
                "start_row": 0,
                "end_row": 69,
                "start_time_s": 0.0,
                "end_time_s": 70.0,
            },
        },
        {
            "segment_index": 1,
            "start": 70,
            "end": 82,
            "phase": "discharge",
            "duration_s": 12.0,
            "sample_count": 12,
            "source_interval": {
                "start_row": 70,
                "end_row": 81,
                "start_time_s": 70.0,
                "end_time_s": 82.0,
            },
        },
    ]
    return frame, py_frame, phases


def test_eligible_pulse_calls_pyprobe_and_maps_r0_r10(monkeypatch) -> None:
    frame, py_frame, phases = _pulse_frames()
    seen = {}

    def fake_get_resistances(result, *, r_times):
        seen["r_times"] = r_times
        seen["soc"] = result.data["SOC"].to_list()
        return SimpleNamespace(
            data=pl.DataFrame(
                {
                    "OCV [V]": [4.0],
                    "SOC": [0.5],
                    "R0 [Ohms]": [0.01],
                    "R_10.0s [Ohms]": [0.02],
                }
            )
        )

    monkeypatch.setattr(response.pulsing, "get_resistances", fake_get_resistances)
    provider_calls = []
    records = response.analyze_pulses(
        frame,
        py_frame,
        phases,
        [
            {
                "cycle_id": 1,
                "complete": True,
                "charge_capacity_ah": 1.0,
                "capacity_coordinate_min_ah": 0.0,
            }
        ],
        config=AnalysisConfig(nominal_capacity_ah=1.0),
        cell_id="cell",
        cycle_id=1,
        provider_calls=provider_calls,
    )

    assert seen["r_times"] == [10.0]
    assert min(seen["soc"]) > 0.9
    assert records[0]["metrics"]["r0"]["value"] == 0.01
    assert records[0]["metrics"]["r_10s"]["value"] == 0.02
    assert provider_calls[0]["status"] == "ok"


def test_pulse_provider_error_has_no_same_name_fallback(monkeypatch) -> None:
    frame, py_frame, phases = _pulse_frames()

    def fail(*args, **kwargs):
        raise RuntimeError("pulse failure")

    monkeypatch.setattr(response.pulsing, "get_resistances", fail)
    records = response.analyze_pulses(
        frame,
        py_frame,
        phases,
        [
            {
                "cycle_id": 1,
                "complete": True,
                "charge_capacity_ah": 1.0,
                "capacity_coordinate_min_ah": 0.0,
            }
        ],
        config=AnalysisConfig(nominal_capacity_ah=1.0),
        cell_id="cell",
        cycle_id=1,
        provider_calls=[],
    )

    assert records[0]["applicability"]["status"] == "not_computable"
    assert records[0]["quality"]["flags"] == ["provider_error"]
    assert records[0]["metrics"] == {}


def test_pulse_without_capacity_reference_does_not_call_pyprobe(monkeypatch) -> None:
    frame, py_frame, phases = _pulse_frames()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("PyProBE must not receive an ungrounded SOC")

    monkeypatch.setattr(response.pulsing, "get_resistances", fail_if_called)
    records = response.analyze_pulses(
        frame,
        py_frame,
        phases,
        [],
        config=AnalysisConfig(),
        cell_id="cell",
        cycle_id=1,
        provider_calls=[],
    )

    assert records[0]["applicability"]["status"] == "not_computable"
    assert records[0]["applicability"]["reasons"] == [
        "no structurally complete cycle capacity coordinate for SOC anchoring"
    ]


def test_rest_provider_error_does_not_emit_local_rest_descriptors(monkeypatch) -> None:
    frame, py_frame, phases = _pulse_frames()

    class FailingExperiment:
        def rest(self):
            raise RuntimeError("rest failure")

    monkeypatch.setattr(response, "pyprobe_experiment", lambda _frame: FailingExperiment())
    provider_calls = []
    records = response.analyze_rest_and_thermal(
        frame,
        py_frame,
        phases,
        config=AnalysisConfig(),
        cell_id="cell",
        cycle_id=1,
        provider_calls=provider_calls,
    )

    assert len(records) == 1
    assert records[0]["metrics"] == {}
    assert records[0]["quality"]["flags"] == ["provider_error"]
    assert provider_calls[0]["status"] == "error"


def test_fractional_relaxation_checkpoint_has_one_stable_key(monkeypatch) -> None:
    time = np.arange(11, dtype=float)
    current = np.r_[np.ones(5), np.zeros(6)]
    voltage = np.r_[np.linspace(3.0, 3.1, 5), np.linspace(3.1, 3.2, 6)]
    frame = pl.DataFrame(
        {
            "_source_row": np.arange(11),
            "test_time_s": time,
            "current_a": current,
            "voltage_v": voltage,
        }
    )
    py_frame = pl.DataFrame(
        {
            "_source_row": np.arange(11),
            "Time [s]": time,
            "Current [A]": current,
            "Voltage [V]": voltage,
            "Cycle": np.ones(11, dtype=int),
            "Step": np.r_[np.zeros(5, dtype=int), np.ones(6, dtype=int)],
            "Event": np.r_[np.zeros(5, dtype=int), np.ones(6, dtype=int)],
            "Capacity [Ah]": np.linspace(0.0, 0.01, 11),
        }
    )
    phases = [
        {
            "segment_index": 0,
            "start": 0,
            "end": 5,
            "phase": "charge",
            "duration_s": 5.0,
            "mode": "constant_current_like",
            "source_interval": {"start_row": 0, "end_row": 4},
        },
        {
            "segment_index": 1,
            "start": 5,
            "end": 11,
            "phase": "rest",
            "previous_phase": "charge",
            "duration_s": 6.0,
            "source_interval": {"start_row": 5, "end_row": 10},
        },
    ]

    class RestExperiment:
        def rest(self):
            return SimpleNamespace(data=py_frame.slice(5, 6))

    monkeypatch.setattr(response, "pyprobe_experiment", lambda _frame: RestExperiment())
    config = AnalysisConfig(relaxation_checkpoints_s=(2.5,))
    records = response.analyze_rest_and_thermal(
        frame,
        py_frame,
        phases,
        config=config,
        cell_id="cell",
        cycle_id=None,
        provider_calls=[],
    )
    summary = analyze_relaxation_summary(
        records,
        config=config,
        cell_id="cell",
        cycle_id=None,
    )

    assert records[0]["metrics"]["voltage_change_at_2.5s"]["status"] == "ok"
    assert "voltage_change_at_2s" not in records[0]["metrics"]
    assert records[0]["metrics"]["duration"]["value"] == 6.0
    assert records[0]["metrics"]["observed_sample_span"]["value"] == 5.0
    assert (
        records[0]["method"]["parameters"]["duration_semantics"]
        == "previous_zoh_support_to_next_phase_boundary"
    )
    assert summary["metrics"]["eligible_rest_count_2.5s"]["value"] == 1


def test_differentiation_error_produces_unavailable_ica_and_dva(monkeypatch) -> None:
    count = 60
    time = np.arange(count, dtype=float)
    voltage = np.linspace(3.0, 4.0, count)
    capacity = np.linspace(0.0, 0.01, count)
    canonical = pl.DataFrame(
        {
            "_source_row": np.arange(count),
            "test_time_s": time,
            "current_a": np.full(count, 0.1),
            "voltage_v": voltage,
            "cycle_index": np.ones(count, dtype=int),
        }
    )
    staged = pl.DataFrame(
        {
            "Time [s]": time,
            "Current [A]": np.full(count, 0.1),
            "Voltage [V]": voltage,
            "Cycle": np.ones(count, dtype=int),
            "Step": np.zeros(count, dtype=int),
            "Event": np.zeros(count, dtype=int),
            "Capacity [Ah]": capacity,
            "_source_row": np.arange(count),
        }
    )
    phases = [
        {
            "segment_index": 0,
            "start": 0,
            "end": count,
            "phase": "charge",
            "duration_s": float(count - 1),
            "sample_count": count,
            "source_interval": {"start_row": 0, "end_row": count - 1},
        }
    ]

    def fail(*args, **kwargs):
        raise RuntimeError("differentiation failure")

    monkeypatch.setattr(response.differentiation, "differentiate_lean", fail)
    records = response.analyze_ica_dva(
        canonical,
        staged,
        phases,
        [],
        config=AnalysisConfig(nominal_capacity_ah=1.0),
        cell_id="cell",
        cycle_id=1,
        provider_calls=[],
    )

    assert {record["record_type"] for record in records} == {
        "response.ica_curve",
        "response.dva_curve",
    }
    assert all(record["quality"]["flags"] == ["provider_error"] for record in records)


def test_no_representative_cycle_skips_ica_dva_providers_even_with_nominal_capacity(
    monkeypatch,
) -> None:
    frame, py_frame, phases = _pulse_frames()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ICA/DVA providers must not run without a representative cycle")

    monkeypatch.setattr(response.differentiation, "differentiate_lean", fail_if_called)
    monkeypatch.setattr(response, "find_peaks", fail_if_called)
    provider_calls = []
    records = response.analyze_ica_dva(
        frame,
        py_frame,
        phases,
        [],
        config=AnalysisConfig(nominal_capacity_ah=1.0),
        cell_id="cell",
        cycle_id=None,
        provider_calls=provider_calls,
    )

    assert [record["record_type"] for record in records] == [
        "response.ica_curve",
        "response.dva_curve",
    ]
    assert all(record["applicability"]["status"] == "not_computable" for record in records)
    assert all(
        record["applicability"]["reasons"] == ["no_structurally_complete_representative_cycle"]
        for record in records
    )
    assert provider_calls == []
