"""Shared synthetic cycler data for BFL 0.4 tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


@pytest.fixture
def vendor_csv(tmp_path: Path) -> Path:
    """Write a BDS-detectable Arbin-style export with ten complete cycles."""

    path = tmp_path / "cell_A.csv"
    fieldnames = [
        "Test Time (s)",
        "Cycle Index",
        "Step Type",
        "Current (A)",
        "Voltage (V)",
        "Aux_Temperature",
    ]
    rows: list[dict[str, float | int | str]] = []
    time_s = 0.0
    for cycle in range(1, 11):
        retention = 1.0 - 0.002 * (cycle - 1)
        phases = (
            ("Rest", 5e-5, 65),
            ("CC_Chg", 0.1 * retention, 100),
            ("Rest", 5e-5, 65),
            ("CC_DChg", -0.1 * retention, 100),
            ("Rest", 5e-5, 65),
        )
        for phase, current, count in phases:
            for index in range(count):
                fraction = index / max(count - 1, 1)
                if phase == "CC_Chg":
                    voltage = 3.1 + 1.0 * fraction
                elif phase == "CC_DChg":
                    voltage = 4.1 - 1.0 * fraction
                else:
                    voltage = 3.7
                rows.append(
                    {
                        "Test Time (s)": time_s,
                        "Cycle Index": cycle,
                        "Step Type": phase,
                        "Current (A)": current,
                        "Voltage (V)": voltage,
                        "Aux_Temperature": 25.0 + 0.01 * cycle,
                    }
                )
                time_s += 1.0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
