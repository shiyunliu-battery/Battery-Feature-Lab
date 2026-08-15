"""Command-line interface for Battery Feature Lab."""

from __future__ import annotations

import argparse
from pathlib import Path

from battery_feature_lab import analyze


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI containing only ``bfl analyze``."""

    parser = argparse.ArgumentParser(prog="bfl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser(
        "analyze", help="Analyze a raw cycler export or a formal BDF artifact."
    )
    command.add_argument("input_path", type=Path)
    command.add_argument("--output-dir", type=Path, default=Path("bfl_outputs"))
    command.add_argument("--input-adapter", choices=("auto", "bds", "bdf"), default="auto")
    command.add_argument("--cell-id")
    command.add_argument("--nominal-capacity-ah", type=float)
    command.add_argument("--representative-cycle", type=int)
    command.add_argument("--declared-protocol-name")
    command.add_argument(
        "--voltage-column",
        help="Exact preprocessed column used as analysis voltage; normalized data is unchanged.",
    )
    command.add_argument(
        "--temperature-column",
        help="Exact preprocessed column used as analysis temperature; normalized data is unchanged.",
    )
    command.add_argument("--formation-cycles-to-exclude", type=int, default=1)
    command.add_argument("--reference-window-size", type=int, default=4)
    command.add_argument(
        "--pulse-resistance-time-s",
        type=float,
        action="append",
        dest="pulse_resistance_times_s",
    )
    command.add_argument(
        "--relaxation-checkpoint-s",
        type=float,
        action="append",
        dest="relaxation_checkpoints_s",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the public CLI."""

    args = build_parser().parse_args(argv)
    result = analyze(
        args.input_path,
        output_dir=args.output_dir,
        input_adapter=args.input_adapter,
        cell_id=args.cell_id,
        nominal_capacity_ah=args.nominal_capacity_ah,
        representative_cycle=args.representative_cycle,
        declared_protocol_name=args.declared_protocol_name,
        formation_cycles_to_exclude=args.formation_cycles_to_exclude,
        reference_window_size=args.reference_window_size,
        pulse_resistance_times_s=tuple(args.pulse_resistance_times_s or (10.0,)),
        relaxation_checkpoints_s=tuple(
            args.relaxation_checkpoints_s or (10.0, 30.0, 60.0, 300.0, 600.0, 1800.0)
        ),
        voltage_column=args.voltage_column,
        temperature_column=args.temperature_column,
    )
    for path in result.files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
