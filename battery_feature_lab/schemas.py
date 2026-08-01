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
            "time_s": (
                "time_s",
                "test_time_s",
                "total_time",
                "total_time_s",
                "time",
                "time_seconds",
                "time/sec",
                "t",
            ),
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
            "capacity_ah": ("capacity_ah", "capacity", "capacity(ah)", "cap_ah", "q"),
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
class EvidenceConfig:
    """Configuration for evidence candidate generation and selection."""

    enabled: bool = True
    question: str | None = None
    token_budget: int = 800
    max_selected_items: int = 12
    min_score: float = 0.0
    token_penalty_cap: float = 0.25
    redundancy_threshold: float = 0.88
    selection_redundancy_weight: float = 0.20
    max_items_per_source: int | None = None
    arithmetic_score_weight: float = 0.65
    geometric_score_weight: float = 0.35
    scoring_weights: dict[str, float] = field(
        default_factory=lambda: {
            "question_relevance": 1.0,
            "lexical_overlap": 1.0,
            "reliability_score": 1.0,
            "protocol_consistency": 1.0,
            "informativeness": 1.0,
            "source_confidence": 1.0,
        }
    )

    def __post_init__(self) -> None:
        if self.token_budget < 0 or self.max_selected_items < 0:
            raise ValueError("Evidence budgets and item limits must be non-negative.")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1.")
        for name in ("token_penalty_cap", "redundancy_threshold", "selection_redundancy_weight"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.max_items_per_source is not None and self.max_items_per_source <= 0:
            raise ValueError("max_items_per_source must be positive when provided.")
        if self.arithmetic_score_weight < 0 or self.geometric_score_weight < 0:
            raise ValueError("Score blend weights must be non-negative.")
        if self.arithmetic_score_weight + self.geometric_score_weight <= 0:
            raise ValueError("At least one score blend weight must be positive.")
        if any(weight < 0 for weight in self.scoring_weights.values()):
            raise ValueError("Scoring component weights must be non-negative.")


@dataclass(frozen=True)
class ProtocolConfig:
    """Configuration for protocol-aware segmentation and recognition.

    Thresholds are deliberately conservative and interpretable. C-rate-relative
    limits are used when ``nominal_capacity_ah`` is provided; otherwise the
    absolute current thresholds apply.
    """

    enabled: bool = True
    # A sample counts as "rest" when |current| is below this (Amps).
    rest_threshold_a: float = 1e-4
    # Within a non-zero-current run, the constant-current part is "constant" when
    # std(|I|)/mean(|I|) stays under this ratio.
    cc_current_cv_tol: float = 0.15
    # The constant-voltage tail is the contiguous suffix where |I| has tapered below
    # this fraction of the run's peak |I| (current decay is the primary CV signal).
    cv_current_fraction: float = 0.8
    # A CV tail must also hold voltage within both of these limits. Current taper
    # alone is not sufficient evidence of constant-voltage operation.
    cv_voltage_tolerance_v: float = 0.02
    cv_voltage_slope_tolerance_v_per_s: float = 2e-4
    # A charge/discharge run adjacent to a rest and no longer than this is a pulse
    # (the GITT/HPPC characterization primitive), in seconds.
    pulse_max_duration_s: float = 3600.0
    # A pulse must be adjacent to a meaningful relaxation, rather than any short
    # zero-current gap. The rest must also be at least this multiple of the pulse.
    pulse_rest_min_duration_s: float = 60.0
    pulse_rest_duration_ratio: float = 1.0
    # Minimum samples for a sub-segment (CC or CV split) to be kept separate.
    min_segment_points: int = 3
    # A cycle is flagged "dynamic" (DST/drive-cycle) when this fraction of its
    # non-rest samples fall in current runs that are neither constant nor a clean taper.
    dynamic_sample_fraction: float = 0.35
    # GITT needs at least this many same-sign pulse+rest repetitions.
    gitt_min_pulses: int = 3
    # HPPC requires repeated pulses in both directions, not merely one pulse of
    # each sign somewhere in the cycle.
    hppc_min_pulses_per_direction: int = 2

    def __post_init__(self) -> None:
        if self.rest_threshold_a < 0:
            raise ValueError("rest_threshold_a must be non-negative.")
        for name in ("cc_current_cv_tol", "cv_current_fraction", "dynamic_sample_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.pulse_max_duration_s <= 0:
            raise ValueError("pulse_max_duration_s must be positive.")
        if self.cv_voltage_tolerance_v < 0:
            raise ValueError("cv_voltage_tolerance_v must be non-negative.")
        if self.cv_voltage_slope_tolerance_v_per_s < 0:
            raise ValueError("cv_voltage_slope_tolerance_v_per_s must be non-negative.")
        if self.pulse_rest_min_duration_s < 0:
            raise ValueError("pulse_rest_min_duration_s must be non-negative.")
        if self.pulse_rest_duration_ratio < 0:
            raise ValueError("pulse_rest_duration_ratio must be non-negative.")
        if self.min_segment_points < 1:
            raise ValueError("min_segment_points must be at least 1.")
        if self.gitt_min_pulses < 1:
            raise ValueError("gitt_min_pulses must be at least 1.")
        if self.hppc_min_pulses_per_direction < 1:
            raise ValueError("hppc_min_pulses_per_direction must be at least 1.")


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
