"""Parquet export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_feature_tables(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    compression: str = "snappy",
) -> dict[str, Path]:
    """Write non-empty feature tables to Parquet files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        if frame is None or frame.empty:
            continue
        path = output / f"{name}.parquet"
        frame.to_parquet(path, index=False, compression=compression)
        paths[name] = path
    return paths
