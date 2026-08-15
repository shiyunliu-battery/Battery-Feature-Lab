"""Run the public BFL contract on local, Git-ignored real datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bfl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=Path, help="One source file or a directory of source files."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-adapter", choices=("auto", "bds", "bdf"), default="auto")
    parser.add_argument("--nominal-capacity-ah", type=float)
    parser.add_argument("--voltage-column")
    parser.add_argument("--temperature-column")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata_files = {"README.md", "MANIFEST.sha256"}
    inputs = (
        [args.input_path]
        if args.input_path.is_file()
        else sorted(
            path
            for path in args.input_path.iterdir()
            if path.is_file() and path.name not in metadata_files
        )
    )
    if not inputs:
        raise SystemExit(f"no local input files found in {args.input_path}")
    failures: list[tuple[Path, str]] = []
    for input_path in inputs:
        try:
            result = bfl.analyze(
                input_path,
                output_dir=args.output_root / input_path.stem,
                input_adapter=args.input_adapter,
                nominal_capacity_ah=args.nominal_capacity_ah,
                voltage_column=args.voltage_column,
                temperature_column=args.temperature_column,
            )
            validation = json.loads(result.analysis_validation_path.read_text(encoding="utf-8"))
            if len(result.files) != 6 or not all(path.is_file() for path in result.files):
                raise RuntimeError("six-file contract was not produced")
            schema_results = validation["schema_validation"]
            invalid = [
                name
                for name, result in schema_results.items()
                if isinstance(result, dict) and result.get("valid") is not True
            ]
            if invalid:
                raise RuntimeError(f"JSON schema validation failed for: {invalid}")
            print(f"PASS {input_path.name}: {validation['record_counts']}")
        except Exception as exc:  # noqa: BLE001 - each dataset failure is reported
            failures.append((input_path, str(exc)))
            print(f"FAIL {input_path.name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
