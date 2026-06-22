"""Data schemas and configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ColumnAliases:
    """Column alias configuration for BDS/cycler exports."""

    aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "cell_id": ("cell_id", "cell", "barcode", "battery_id", "channel_id"),
            "cycle_index": ("cycle_index", "cycle", "cycle_number", "cycle_number_index"),
            "step_index": ("step_index", "step", "step_number", "step_id"),
            "step_type": ("step_type", "state", "mode", "step_name", "status"),
            "time_s": ("time_s", "test_time_s", "time", "time_seconds", "time/sec", "t"),
            "voltage_v": ("voltage_v", "voltage", "voltage(v)", "ewe/v", "v"),
            "current_a": ("current_a", "current", "current(a)", "i/a", "i"),
            "temperature_c": (
                "temperature_c",
                "temperature",
                "temperature(c)",
                "temp_c",
                "aux_temperature",
            ),
            "charge_capacity_ah": (
                "charge_capacity_ah",
                "charge_capacity",
                "charge_cap_ah",
                "qc",
                "q_charge",
            ),
            "discharge_capacity_ah": (
                "discharge_capacity_ah",
                "discharge_capacity",
                "discharge_cap_ah",
                "qd",
                "q_discharge",
            ),
            "energy_wh": ("energy_wh", "energy", "energy(w.h)", "wh"),
            "soc": ("soc", "state_of_charge", "soc_percent"),
            "frequency_hz": ("frequency_hz", "freq_hz", "frequency", "freq"),
            "z_real_ohm": ("z_real_ohm", "zreal", "re_z", "z_real", "z'"),
            "z_imag_ohm": ("z_imag_ohm", "zimag", "im_z", "z_imag", "z''"),
        }
    )


@dataclass(frozen=True)
class ReaderConfig:
    """Configuration used when importing raw data."""

    cell_id: str | None = None
    column_aliases: ColumnAliases = field(default_factory=ColumnAliases)
    positive_current_is_charge: bool = True
    current_rest_threshold_a: float = 1e-4
    time_unit: str = "s"
    capacity_unit: str = "Ah"
    soc_unit: str = "fraction"


@dataclass(frozen=True)
class FeatureConfig:
    """Feature extraction configuration."""

    nominal_capacity_ah: float | None = None
    voltage_grid_points: int = 1000
    capacity_grid_points: int = 1000
    smoothing_window: int = 21
    smoothing_polyorder: int = 3
    early_reference_cycle: int = 10
    early_target_cycle: int = 100
    delta_q_voltage_points: int = 1000
    histogram_bins: int = 20
    max_peaks: int = 5
    min_points_per_cycle: int = 8
    min_points_for_curve: int = 25
    peak_prominence_noise_multiplier: float = 6.0
    # SOC level above which rest counts as "high-SOC exposure". Calendar-aging studies
    # (Schmalstieg et al., J. Power Sources 2014; Keil et al., J. Electrochem. Soc. 2016)
    # show aging accelerates markedly above ~80% SOC / high electrode potential.
    high_soc_level: float = 0.8


@dataclass(frozen=True)
class DiagnosticConfig:
    """Configuration for rule-based diagnostic evidence tagging."""

    trend_p_value_alpha: float = 0.05
    # Mann-Kendall has low power on very short series; the trend test is conventionally applied at
    # n>=8-10. Default to 8 so reported trend tags are statistically meaningful.
    min_trend_points: int = 8
    # Batch-relative outlier parameters (supplementary; require a population of cells).
    stress_percentile: float = 0.9
    # Robust outlier cutoff in MAD-scaled sigmas above the median (median + k * 1.4826 * MAD).
    # MAD is used instead of mean+k*std because a single extreme cell inflates the standard
    # deviation and can mask itself; the standard robust outlier rule uses k=3.
    stress_mad_threshold: float = 3.0
    # Absolute, single-cell-capable threshold for high-SOC rest exposure. The SOC *level*
    # that counts as "high" is grounded in calendar-aging literature (see FeatureConfig.
    # high_soc_level); this *fraction-of-time* default is a conservative engineering
    # convention (flag when the majority of observed time is spent resting at high SOC),
    # not a measured physical constant — tune per application or set None to disable.
    high_soc_rest_fraction_threshold: float | None = 0.5
    # No universal absolute threshold exists for C-rate variance; leave None to rely on the
    # batch-relative outlier criterion, or inject a domain-specific value for single-cell use.
    c_rate_variance_threshold: float | None = None
    # Absolute, datasheet-derived limit. Only fires when the cell specification is provided.
    datasheet_max_discharge_c_rate: float | None = None
    max_discharge_c_rate_fraction: float = 0.9


@dataclass(frozen=True)
class ExportConfig:
    """Feature export configuration."""

    output_dir: Path
    write_normalized_timeseries: bool = True
    parquet_compression: str = "snappy"


@dataclass(frozen=True)
class FeatureTable:
    """Named feature table returned by extractors."""

    name: str
    frame: Any
