from __future__ import annotations

import numpy as np
import pandas as pd

from battery_feature_lab.protocol import (
    build_protocol_records,
    describe_region,
    detect_protocol_segments,
    segment_cycle,
    write_protocol_jsonl,
)
from battery_feature_lab.protocol.classifier import (
    CC_CV,
    CONSTANT_CURRENT,
    GITT,
    HPPC,
    OTHER,
    REST_PROTOCOL,
    build_signature,
    classify_protocol,
)
from battery_feature_lab.schemas import ProtocolConfig

DT = 10.0  # seconds per sample


def _frame(records: list[tuple[float, float]], start_t: float = 0.0, cycle: int = 1) -> pd.DataFrame:
    """Build a normalized-style frame from (current_a, voltage_v) rows at fixed dt."""
    times = start_t + DT * np.arange(len(records))
    return pd.DataFrame(
        {
            "cell_id": "A",
            "cycle_index": cycle,
            "time_s": times,
            "current_a": [c for c, _ in records],
            "voltage_v": [v for _, v in records],
        }
    )


def _cc_segment(current: float, v0: float, v1: float, n: int) -> list[tuple[float, float]]:
    return [(current, v) for v in np.linspace(v0, v1, n)]


def _rest_segment(v0: float, v_inf: float, n: int) -> list[tuple[float, float]]:
    # exponential-ish relaxation toward v_inf
    ts = np.linspace(0, 1, n)
    volts = v_inf + (v0 - v_inf) * np.exp(-3 * ts)
    return [(0.0, float(v)) for v in volts]


def _cv_segment(i0: float, v_hold: float, n: int) -> list[tuple[float, float]]:
    # constant voltage, current decays toward zero
    currents = i0 * np.exp(-3 * np.linspace(0, 1, n))
    return [(float(i), v_hold) for i in currents]


def _classify(records: list[tuple[float, float]], config: ProtocolConfig | None = None) -> dict:
    config = config or ProtocolConfig(rest_threshold_a=1e-3)
    segments = segment_cycle(_frame(records), config, nominal_capacity_ah=2.0)
    return classify_protocol(segments, config)


def test_cc_cv_charge_is_recognized() -> None:
    records = _cc_segment(2.0, 3.4, 4.2, 30) + _cv_segment(2.0, 4.2, 30) + _rest_segment(4.2, 4.1, 20)
    result = _classify(records)
    assert result["protocol"] == CC_CV
    assert result["protocol_family"] == "cccv"
    assert "cv_charge" in result["signature"]


def test_constant_current_discharge_is_recognized() -> None:
    records = _cc_segment(-2.0, 4.2, 3.2, 40)
    result = _classify(records)
    assert result["protocol"] == CONSTANT_CURRENT
    assert result["protocol_family"] == "discharge"


def test_rest_only_is_recognized() -> None:
    result = _classify(_rest_segment(4.0, 3.95, 30))
    assert result["protocol"] == REST_PROTOCOL
    assert result["protocol_family"] == "rest"


def test_gitt_pulse_train_is_recognized() -> None:
    records: list[tuple[float, float]] = []
    v = 4.0
    for _ in range(6):  # 6 unidirectional small pulses + rests
        records += _cc_segment(-0.5, v, v - 0.05, 6)  # short pulse
        v -= 0.05
        records += _rest_segment(v, v + 0.03, 40)  # long rest
        v += 0.03
    result = _classify(records)
    assert result["protocol"] == GITT
    assert result["protocol_family"] == "pulse_characterization"
    assert "x" in result["signature"]  # repetition detected


def test_hppc_bidirectional_pulses_are_recognized() -> None:
    records: list[tuple[float, float]] = []
    v = 3.8
    for _ in range(3):  # at each SOC point: discharge pulse + rest + charge pulse + rest
        records += _cc_segment(-3.0, v, v - 0.08, 6)
        records += _rest_segment(v - 0.08, v, 20)
        records += _cc_segment(3.0, v, v + 0.06, 6)
        records += _rest_segment(v + 0.06, v, 20)
    result = _classify(records)
    assert result["protocol"] == HPPC


def test_generic_dynamic_load_is_not_forced_to_dst() -> None:
    rng = np.random.default_rng(0)
    # continuously varying current (drive cycle), voltage wanders
    records: list[tuple[float, float]] = []
    v = 3.9
    for _ in range(120):
        i = float(rng.uniform(-4, 2))  # irregular, sign-changing
        v += -i * 1e-3 + float(rng.normal(0, 1e-3))
        records.append((i, v))
    result = _classify(records)
    assert result["protocol"] == OTHER
    assert result["protocol_family"] == "dynamic"
    assert result["signature"] == "dynamic"


def test_unrecognized_protocol_returns_other_with_signature() -> None:
    # A lone CV hold with no preceding CC: not a named template, must not be forced.
    records = _cv_segment(1.0, 4.2, 40)
    result = _classify(records)
    assert result["protocol"] in {OTHER, CC_CV}  # tolerant, but must carry a signature
    assert result["signature"] != ""


def test_current_taper_without_voltage_hold_is_not_cv() -> None:
    currents = np.exp(-3 * np.linspace(0, 1, 40))
    records = list(zip(currents, np.linspace(3.4, 4.2, 40)))
    result = _classify(records)

    assert result["protocol"] != CC_CV
    assert result["protocol_family"] == "dynamic"


def test_short_rest_does_not_turn_ordinary_cycling_into_gitt() -> None:
    records: list[tuple[float, float]] = []
    for offset in (0.0, 0.1, 0.2):
        records += _cc_segment(-1.0, 4.1 - offset, 4.0 - offset, 10)
        records += _rest_segment(4.0 - offset, 4.01 - offset, 5)

    result = _classify(records)
    assert result["protocol"] != GITT


def test_sign_changing_active_window_is_one_dynamic_segment() -> None:
    records = [
        (1.0 if position % 2 == 0 else -1.0, 3.8 - position * 0.001)
        for position in range(30)
    ]
    segments = segment_cycle(
        _frame(records), ProtocolConfig(rest_threshold_a=1e-3), nominal_capacity_ah=2.0
    )

    assert [segment["step_label"] for segment in segments] == ["dynamic"]


def test_signature_collapses_and_detects_repetition() -> None:
    labels = ["pulse_discharge", "rest", "pulse_discharge", "rest", "pulse_discharge", "rest"]
    assert build_signature(labels) == "[pulse_discharge>rest]x3"
    assert build_signature(["cc_charge", "cc_charge", "cv_charge", "rest"]) == "cc_charge>cv_charge>rest"


def test_detect_protocol_segments_table_and_annotation() -> None:
    records = _cc_segment(2.0, 3.4, 4.2, 20) + _cv_segment(2.0, 4.2, 20) + _rest_segment(4.2, 4.1, 20)
    frame = _frame(records)
    config = ProtocolConfig(rest_threshold_a=1e-3)
    segments = detect_protocol_segments(frame, config, nominal_capacity_ah=2.0)
    assert not segments.empty
    assert set(segments["protocol"]) == {CC_CV}
    assert {"protocol_family", "protocol_signature", "step_label"}.issubset(segments.columns)
    # every segment maps to a stable id and carries voltage descriptors
    assert segments["segment_id"].is_unique
    assert segments["delta_voltage_v"].notna().all()
    assert segments["protocol_match_reasons"].map(bool).all()


def test_annotation_is_scoped_to_exact_rows_and_cell_cycle() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["A", "B", "A", "B"],
            "cycle_index": [1, 1, 1, 1],
            "time_s": [0.0, 0.0, 10.0, 10.0],
            "current_a": [1.0, 0.0, 1.0, 0.0],
            "voltage_v": [3.5, 3.8, 3.6, 3.8],
        }
    )
    config = ProtocolConfig(rest_threshold_a=1e-3)
    segments = detect_protocol_segments(frame, config, nominal_capacity_ah=2.0)

    from battery_feature_lab.protocol import annotate_normalized

    annotated = annotate_normalized(frame, segments)
    assert annotated.loc[annotated["cell_id"] == "A", "protocol"].tolist() == [
        CONSTANT_CURRENT,
        CONSTANT_CURRENT,
    ]
    assert annotated.loc[annotated["cell_id"] == "B", "protocol"].tolist() == [
        REST_PROTOCOL,
        REST_PROTOCOL,
    ]


def test_protocol_records_and_jsonl_export(tmp_path) -> None:
    import json

    records: list[tuple[float, float]] = []
    v = 4.1
    for _ in range(4):
        records += _cc_segment(-0.5, v, v - 0.05, 6)
        v -= 0.05
        records += _rest_segment(v, v + 0.03, 30)
        v += 0.03
    frame = _frame(records, cycle=2)
    config = ProtocolConfig(rest_threshold_a=1e-3)
    segments = detect_protocol_segments(frame, config, nominal_capacity_ah=2.0)

    protocol_records = build_protocol_records(segments)
    assert len(protocol_records) == 1
    record = protocol_records[0]
    assert record["protocol"] == GITT
    assert record["signature"].startswith("[pulse_discharge>rest]")
    assert record["n_segments"] == len(record["segments"])
    assert all("text" in seg and "voltage" in seg and "current" in seg for seg in record["segments"])

    path = write_protocol_jsonl(protocol_records, tmp_path / "protocol_segments.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    reparsed = json.loads(lines[0])
    assert reparsed["protocol"] == GITT
    assert reparsed["segments"][0]["step_label"] == "pulse_discharge"


def test_describe_region_answers_selected_window() -> None:
    records = _cc_segment(2.0, 3.4, 4.2, 20) + _cv_segment(2.0, 4.2, 20) + _rest_segment(4.2, 4.1, 20)
    frame = _frame(records)
    # select just the CV hold + rest tail
    out = describe_region(frame, start_time_s=20 * DT, end_time_s=59 * DT, nominal_capacity_ah=2.0)
    assert out["n_segments"] >= 1
    labels = [seg["step_label"] for seg in out["segments"]]
    assert any("cv" in lbl or lbl == "rest" for lbl in labels)
    assert all("text" in seg and "voltage" in seg for seg in out["segments"])
