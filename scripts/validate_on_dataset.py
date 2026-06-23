"""Validate early-life feature effectiveness against observed cycle life.

Two modes:

  # Offline self-test: synthesize a population whose cells have a *known* life-vs-curve
  # relationship, and confirm the harness recovers the expected negative correlation
  # between log(var Delta Q(V)) and log(cycle life). Needs no download.
  python scripts/validate_on_dataset.py --synthetic 12

  # Real dataset: a folder of per-cell cycler/BDS CSVs (one cell per file). Column names are
  # normalized by the BDS reader, so common cycler aliases are accepted.
  python scripts/validate_on_dataset.py --data-dir path/to/cells --nominal-capacity-ah 1.1
Export each cell to a CSV with the canonical columns (see README) before running --data-dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a plain script (python scripts/validate_on_dataset.py) without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from battery_feature_lab.bds_adapter.readers import read_bds_export  # noqa: E402
from battery_feature_lab.schemas import FeatureConfig, ReaderConfig  # noqa: E402
from battery_feature_lab.validation import validate_early_life_features  # noqa: E402


def generate_synthetic_population(
    n_cells: int = 12, n_cycles: int = 460, seed: int = 0
) -> pd.DataFrame:
    """Synthesize cells with a known severity parameter driving both fade rate and the
    voltage-localized curve deformation that Delta Q(V) variance is designed to capture.

    Higher severity -> faster fade (shorter life) AND a deeper localized dip developing in the
    discharge V-Q curve by cycle 100 -> larger var(Delta Q(100-10)). The harness should therefore
    recover a strong NEGATIVE correlation between log(var Delta Q) and log(life). This is a
    self-test of the machinery, not a scientific result.
    """

    rng = np.random.default_rng(seed)
    nominal = 1.1
    severities = np.linspace(0.05, 1.0, n_cells)
    n_charge, n_disch, n_rest = 22, 40, 4
    frames: list[pd.DataFrame] = []

    for idx, s in enumerate(severities):
        fade_per_cycle = 0.000444 * (1.0 + 2.0 * s)  # life ~ 0.2/fade spans ~150..450 cycles
        dip_amp = 0.05 * s  # voltage-localized capacity-loss feature, grows with severity
        cell_id = f"syn_{idx:02d}"
        t = 0.0
        cell_rows: list[dict[str, object]] = []
        for cycle in range(1, n_cycles + 1):
            qmax = nominal * max(0.05, 1.0 - fade_per_cycle * cycle)
            noise = rng.normal(0, 0.0015)

            # charge
            qc = np.linspace(0, qmax, n_charge)
            vc = 3.0 + 0.55 * (qc / qmax) + 0.01 * np.sin(qc / qmax * np.pi) + noise
            for q, v in zip(qc, vc):
                cell_rows.append(_row(cell_id, cycle, 0, t, v, 1.1, qmax, q, 0.0))
                t += 30

            # rest at top of charge (high SOC)
            for v in np.linspace(vc[-1], vc[-1] - 0.03, n_rest):
                cell_rows.append(_row(cell_id, cycle, 1, t, v, 0.0, qmax, qmax, 0.0))
                t += 20

            # discharge with a severity/aging-dependent localized dip in the V-Q curve
            x = np.linspace(0, 0.997, n_disch)  # fractional capacity
            deform = dip_amp * (cycle / 100.0) * np.exp(-(((x - 0.5) / 0.12) ** 2))
            vd = 3.45 - 0.6 * x - deform + noise
            qd = x * qmax
            for q, v in zip(qd, vd):
                cell_rows.append(_row(cell_id, cycle, 2, t, v, -1.1, qmax, 0.0, q))
                t += 30
        frames.append(pd.DataFrame(cell_rows))

    return pd.concat(frames, ignore_index=True)


def _row(cell_id, cycle, step, t, v, i, qmax, qc, qd):
    return {
        "cell_id": cell_id,
        "cycle_index": cycle,
        "step_index": step,
        "step_type": {0: "charge", 1: "rest", 2: "discharge"}[step],
        "time_s": t,
        "voltage_v": v,
        "current_a": i,
        "temperature_c": 25.0,
        "charge_capacity_ah": qc,
        "discharge_capacity_ah": qd,
    }


def load_cell_folder(data_dir: Path, nominal_capacity_ah: float | None) -> pd.DataFrame:
    """Read every CSV in a folder as one cell and concatenate normalized frames."""

    frames = []
    for path in sorted(data_dir.glob("*.csv")):
        normalized = read_bds_export(path, ReaderConfig(cell_id=path.stem))
        if "cell_id" not in normalized.columns or normalized["cell_id"].isna().all():
            normalized["cell_id"] = path.stem
        frames.append(normalized)
    if not frames:
        raise SystemExit(f"No CSV files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate early-life features vs cycle life.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", type=int, metavar="N_CELLS", help="Run the offline self-test.")
    group.add_argument("--data-dir", type=Path, help="Folder of per-cell CSVs.")
    parser.add_argument("--nominal-capacity-ah", type=float, default=1.1)
    parser.add_argument("--eol-fraction", type=float, default=0.8)
    parser.add_argument("--reference-cycle", type=int, default=10)
    parser.add_argument("--target-cycle", type=int, default=100)
    parser.add_argument("--top", type=int, default=10, help="How many ranked features to show.")
    args = parser.parse_args(argv)

    if args.synthetic is not None:
        normalized = generate_synthetic_population(n_cells=args.synthetic)
    else:
        normalized = load_cell_folder(args.data_dir, args.nominal_capacity_ah)

    config = FeatureConfig(
        nominal_capacity_ah=args.nominal_capacity_ah,
        early_reference_cycle=args.reference_cycle,
        early_target_cycle=args.target_cycle,
    )
    result = validate_early_life_features(
        normalized, config=config, eol_fraction=args.eol_fraction
    )

    print(f"Cells total: {result['n_cells_total']} | uncensored (reached EOL): {result['n_cells_uncensored']}")
    headline = result["delta_q_headline"]
    if headline:
        print("\nHeadline check (var Delta Q(V) vs life):")
        print(
            f"  log(var Delta Q) vs log10(life): n={headline['n_cells']}  "
            f"Pearson r={headline['pearson_r']:+.3f}  R^2={headline['r2']:.3f}  "
            f"Spearman rho={headline['spearman_rho']:+.3f}"
        )
        ok = headline["pearson_r"] < -0.5
        print(f"  Expected strong NEGATIVE correlation: {'PASS' if ok else 'WEAK/UNEXPECTED'}")
    else:
        print("\nNot enough uncensored cells to compute the headline statistic.")

    report = result["correlation_report"]
    if not report.empty:
        print(f"\nTop {args.top} early-life features by |Spearman| with log10(cycle life):")
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(report.head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
