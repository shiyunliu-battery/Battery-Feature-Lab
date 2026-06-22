"""Command-line interface for Battery Feature Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from battery_feature_lab.pipeline import FeaturePipeline, PipelineConfig
from battery_feature_lab.schemas import DiagnosticConfig, ExportConfig, FeatureConfig, ReaderConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="battery-features")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract battery features from a BDS export.")
    extract.add_argument("input_path", type=Path)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--cell-id", type=str, default=None)
    extract.add_argument("--nominal-capacity-ah", type=float, default=None)
    extract.add_argument("--negative-current-is-charge", action="store_true")
    extract.add_argument("--time-unit", choices=["s", "min", "h"], default="s")
    extract.add_argument("--capacity-unit", choices=["Ah", "mAh"], default="Ah")
    extract.add_argument("--soc-unit", choices=["fraction", "percent"], default="fraction")
    extract.add_argument("--reference-cycle", type=int, default=10)
    extract.add_argument("--target-cycle", type=int, default=100)
    extract.add_argument("--histogram-bins", type=int, default=20)
    extract.add_argument("--trend-p-value-alpha", type=float, default=0.05)
    extract.add_argument("--stress-percentile", type=float, default=0.9)
    extract.add_argument("--stress-mad-threshold", type=float, default=3.0)
    extract.add_argument("--high-soc-level", type=float, default=0.8)
    extract.add_argument("--high-soc-rest-threshold", type=float, default=None)
    extract.add_argument("--c-rate-variance-threshold", type=float, default=None)
    extract.add_argument("--datasheet-max-discharge-c-rate", type=float, default=None)
    extract.add_argument("--peak-prominence-noise-multiplier", type=float, default=6.0)
    extract.add_argument("--skip-normalized-timeseries", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        # Only override the config defaults when the user supplied a value, so the documented
        # built-in defaults (e.g. the single-cell high-SOC-rest threshold) are preserved.
        diagnostic_kwargs: dict[str, float] = {
            "trend_p_value_alpha": args.trend_p_value_alpha,
            "stress_percentile": args.stress_percentile,
            "stress_mad_threshold": args.stress_mad_threshold,
        }
        if args.high_soc_rest_threshold is not None:
            diagnostic_kwargs["high_soc_rest_fraction_threshold"] = args.high_soc_rest_threshold
        if args.c_rate_variance_threshold is not None:
            diagnostic_kwargs["c_rate_variance_threshold"] = args.c_rate_variance_threshold
        if args.datasheet_max_discharge_c_rate is not None:
            diagnostic_kwargs["datasheet_max_discharge_c_rate"] = args.datasheet_max_discharge_c_rate
        config = PipelineConfig(
            reader=ReaderConfig(
                cell_id=args.cell_id,
                positive_current_is_charge=not args.negative_current_is_charge,
                time_unit=args.time_unit,
                capacity_unit=args.capacity_unit,
                soc_unit=args.soc_unit,
            ),
            features=FeatureConfig(
                nominal_capacity_ah=args.nominal_capacity_ah,
                early_reference_cycle=args.reference_cycle,
                early_target_cycle=args.target_cycle,
                histogram_bins=args.histogram_bins,
                peak_prominence_noise_multiplier=args.peak_prominence_noise_multiplier,
                high_soc_level=args.high_soc_level,
            ),
            export=ExportConfig(
                output_dir=args.output_dir,
                write_normalized_timeseries=not args.skip_normalized_timeseries,
            ),
            diagnostics=DiagnosticConfig(**diagnostic_kwargs),
        )
        tables = FeaturePipeline(config).run(args.input_path)
        non_empty = {name: len(frame) for name, frame in tables.items() if frame is not None and not frame.empty}
        print(f"Wrote features to {args.output_dir}")
        print(f"Non-empty tables: {non_empty}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
