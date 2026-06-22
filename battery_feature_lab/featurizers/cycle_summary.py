"""Per-cycle summary feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_feature_lab.core.cycle_splitter import iter_cycles
from battery_feature_lab.core.integration import (
    capacity_from_columns_or_integral,
    integrate_energy_wh,
    trapezoid,
)
from battery_feature_lab.featurizers.base import BaseFeaturizer
from battery_feature_lab.featurizers.common import describe_array, duration_s, safe_ratio
from battery_feature_lab.schemas import FeatureTable


class CycleSummaryFeaturizer(BaseFeaturizer):
    """Extract capacity, energy, voltage, current and temperature summaries per cycle."""

    name = "cycle_features"

    def extract(self, frame: pd.DataFrame) -> FeatureTable:
        rows: list[dict[str, float | int | str]] = []
        for cell_id, cycle_index, cycle in iter_cycles(frame):
            if len(cycle) < self.config.min_points_per_cycle:
                continue
            charge = cycle[cycle["step_type"] == "charge"]
            discharge = cycle[cycle["step_type"] == "discharge"]
            rest = cycle[cycle["step_type"] == "rest"]

            charge_capacity = capacity_from_columns_or_integral(charge, "charge_capacity_ah")
            discharge_capacity = capacity_from_columns_or_integral(
                discharge, "discharge_capacity_ah"
            )
            charge_energy = integrate_energy_wh(charge)
            discharge_energy = integrate_energy_wh(discharge)

            current_abs = cycle["current_a"].abs()
            nominal = self.config.nominal_capacity_ah
            c_rate = current_abs / nominal if nominal else pd.Series(np.nan, index=cycle.index)

            charge_voltage = charge["voltage_v"].to_numpy(dtype=float)
            cv_capacity = _estimate_cv_capacity_fraction(charge)

            row: dict[str, float | int | str] = {
                "cell_id": cell_id,
                "cycle_index": cycle_index,
                "n_points": int(len(cycle)),
                "duration_s": duration_s(cycle),
                "charge_duration_s": duration_s(charge),
                "discharge_duration_s": duration_s(discharge),
                "rest_duration_s": duration_s(rest),
                "charge_capacity_ah": charge_capacity,
                "discharge_capacity_ah": discharge_capacity,
                "charge_energy_wh": charge_energy,
                "discharge_energy_wh": discharge_energy,
                "coulombic_efficiency": safe_ratio(discharge_capacity, charge_capacity),
                "energy_efficiency": safe_ratio(discharge_energy, charge_energy),
                "specific_throughput_efc": safe_ratio(
                    0.5 * (charge_capacity + discharge_capacity), nominal or float("nan")
                ),
                "cv_capacity_fraction": cv_capacity,
                "charge_voltage_max_v": float(np.nanmax(charge_voltage))
                if len(charge_voltage)
                else float("nan"),
                "charge_voltage_min_v": float(np.nanmin(charge_voltage))
                if len(charge_voltage)
                else float("nan"),
            }
            row.update(describe_array("voltage_v", cycle["voltage_v"].to_numpy(dtype=float)))
            row.update(describe_array("current_a", cycle["current_a"].to_numpy(dtype=float)))
            row.update(describe_array("abs_current_a", current_abs.to_numpy(dtype=float)))
            row.update(describe_array("c_rate", c_rate.to_numpy(dtype=float)))
            if "temperature_c" in cycle.columns:
                row.update(describe_array("temperature_c", cycle["temperature_c"].to_numpy(dtype=float)))
                row["temperature_integral_c_s"] = _temperature_integral(cycle)
            rows.append(row)
        return FeatureTable(self.name, pd.DataFrame(rows))


def _estimate_cv_capacity_fraction(charge: pd.DataFrame) -> float:
    """Estimate CV capacity fraction from near-maximum voltage charging points."""

    if charge.empty or "voltage_v" not in charge.columns:
        return float("nan")
    vmax = charge["voltage_v"].max()
    if not np.isfinite(vmax):
        return float("nan")
    cv = charge[charge["voltage_v"] >= vmax - 0.01]
    total = capacity_from_columns_or_integral(charge, "charge_capacity_ah")
    cv_cap = capacity_from_columns_or_integral(cv, "charge_capacity_ah")
    return safe_ratio(cv_cap, total)


def _temperature_integral(cycle: pd.DataFrame) -> float:
    values = cycle[["time_s", "temperature_c"]].dropna()
    if len(values) < 2:
        return float("nan")
    time = values["time_s"].to_numpy(dtype=float)
    temp = values["temperature_c"].to_numpy(dtype=float)
    return trapezoid(temp, time)
