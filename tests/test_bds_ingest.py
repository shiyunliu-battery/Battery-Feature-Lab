"""Tests for the BDS-backed ingest layer."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import bfl
from battery_feature_lab.bds_adapter.readers import (
    read_bds_export,
    read_bds_export_with_report,
)
from battery_feature_lab.schemas import ReaderConfig


def _vendor_frame(cycles: int = 4) -> pd.DataFrame:
    """Build an export using vendor header spellings, not BFL internal names."""

    rows = []
    for cycle in range(1, cycles + 1):
        for phase, current, count in (
            ("CC_Chg", 1.1, 60),
            ("Rest", 0.0, 10),
            ("CC_DChg", -1.1, 60),
            ("Rest", 0.0, 10),
        ):
            for k in range(count):
                frac = k / max(count - 1, 1)
                if phase == "CC_Chg":
                    voltage = 3.2 + 0.9 * frac
                elif phase == "CC_DChg":
                    voltage = 4.1 - 0.9 * frac
                else:
                    voltage = 3.7
                rows.append(
                    {
                        "Cycle Index": cycle,
                        "Step Type": phase,
                        "Current (A)": current,
                        "Voltage (V)": voltage,
                        "Aux_Temperature": 25.0,
                        "SOC": 100.0 * frac,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.insert(0, "Test Time (s)", np.arange(len(frame), dtype=float))
    return frame


def test_reads_vendor_headers_through_bds(tmp_path):
    """BDS resolves header spellings that BFL's own alias table never covered."""

    path = tmp_path / "cell_A.csv"
    _vendor_frame().to_csv(path, index=False)

    frame = read_bds_export(path)

    assert {"time_s", "voltage_v", "current_a", "cell_id", "step_type"} <= set(frame.columns)
    assert frame["cell_id"].eq("cell_A").all()
    assert set(frame["step_type"].unique()) == {"charge", "discharge", "rest"}
    assert frame["cycle_index"].nunique() == 4
    assert frame["time_s"].is_monotonic_increasing


def test_passthrough_columns_survive_the_bds_handoff(tmp_path):
    """Columns BDS does not model must still reach the featurizers."""

    path = tmp_path / "cell_B.csv"
    _vendor_frame(cycles=2).to_csv(path, index=False)

    frame = read_bds_export(path)

    assert "soc" in frame.columns
    assert "temperature_c" in frame.columns
    assert frame["temperature_c"].notna().all()


def test_json_and_csv_agree(tmp_path):
    """JSON is staged to CSV so both routes share one normalization path."""

    source = _vendor_frame(cycles=2)
    csv_path = tmp_path / "cell_C.csv"
    jsonl_path = tmp_path / "cell_C.jsonl"
    source.to_csv(csv_path, index=False)
    source.to_json(jsonl_path, orient="records", lines=True)

    from_csv = read_bds_export(csv_path)
    from_jsonl = read_bds_export(jsonl_path)

    assert list(from_csv.columns) == list(from_jsonl.columns)
    np.testing.assert_allclose(from_csv["voltage_v"], from_jsonl["voltage_v"])


def test_report_records_adapter_provenance(tmp_path):
    """The BDS conversion report is the audit trail for a feature run."""

    path = tmp_path / "cell_D.csv"
    _vendor_frame(cycles=2).to_csv(path, index=False)

    _, report = read_bds_export_with_report(path)

    assert report["cycler"]
    assert report["schema_version"]


def test_run_metadata_embeds_the_bds_report(tmp_path):
    path = tmp_path / "cell_E.csv"
    _vendor_frame(cycles=4).to_csv(path, index=False)

    bfl.extract(path, output_dir=tmp_path / "out", nominal_capacity_ah=1.1)
    metadata = json.loads((tmp_path / "out" / "run_metadata.json").read_text())

    assert metadata["reader_config"]["ingest"] == "battery-data-standard"
    assert metadata["bds_conversion_report"]["cycler"]


def test_unreadable_file_points_at_bds_doctor(tmp_path):
    path = tmp_path / "not_cycler_data.csv"
    path.write_text("alpha,beta\n1,2\n3,4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bds doctor"):
        read_bds_export(path)


def test_deprecated_sign_flag_warns(tmp_path):
    path = tmp_path / "cell_F.csv"
    _vendor_frame(cycles=2).to_csv(path, index=False)

    with pytest.warns(DeprecationWarning, match="positive_current_is_charge"):
        read_bds_export(path, ReaderConfig(positive_current_is_charge=False))
