"""Shared constants used across the feature pipeline."""

CANONICAL_COLUMNS = {
    "cell_id",
    "cycle_index",
    "step_index",
    "step_type",
    "time_s",
    "voltage_v",
    "current_a",
    "temperature_c",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "energy_wh",
    "soc",
    "frequency_hz",
    "z_real_ohm",
    "z_imag_ohm",
}

STEP_CHARGE = "charge"
STEP_DISCHARGE = "discharge"
STEP_REST = "rest"

EPS = 1e-12
