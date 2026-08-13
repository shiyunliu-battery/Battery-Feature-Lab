"""Conservative EIS descriptors.

The historical module/class name is retained for import compatibility, but BFL does
not perform DRT inversion and does not emit DRT peak placeholders. The outputs are
strictly descriptive impedance-curve features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_feature_lab.featurizers.base import BaseFeaturizer
from battery_feature_lab.featurizers.common import describe_array
from battery_feature_lab.schemas import FeatureTable


class EISDRTFeaturizer(BaseFeaturizer):
    """Legacy-named compatibility class that extracts EIS descriptors only."""

    name = "eis_features"

    def extract(self, frame: pd.DataFrame) -> FeatureTable:
        required = {"frequency_hz", "z_real_ohm", "z_imag_ohm"}
        if not required.issubset(frame.columns):
            return FeatureTable(self.name, pd.DataFrame())
        rows = []
        group_cols = ["cell_id"] + (["cycle_index"] if "cycle_index" in frame.columns else [])
        for key, group in frame.groupby(group_cols, sort=True):
            features = _eis_features(group)
            if not features:
                continue
            if isinstance(key, tuple):
                features["cell_id"] = key[0]
                if len(key) > 1:
                    features["cycle_index"] = int(key[1])
            else:
                features["cell_id"] = key
            rows.append(features)
        return FeatureTable(self.name, pd.DataFrame(rows))


def _eis_features(group: pd.DataFrame) -> dict[str, float]:
    data = group[["frequency_hz", "z_real_ohm", "z_imag_ohm"]].dropna().sort_values("frequency_hz")
    if data.empty:
        return {}
    freq = data["frequency_hz"].to_numpy(dtype=float)
    z_real = data["z_real_ohm"].to_numpy(dtype=float)
    z_imag = data["z_imag_ohm"].to_numpy(dtype=float)
    z_abs = np.sqrt(z_real**2 + z_imag**2)
    high_freq_idx = int(np.nanargmax(freq))
    min_imag_idx = int(np.nanargmin(np.abs(z_imag)))
    features = {
        "eis_point_count": int(len(data)),
        "eis_z_real_at_max_frequency_ohm": float(z_real[high_freq_idx]),
        "eis_z_real_at_min_abs_imag_ohm": float(z_real[min_imag_idx]),
        "eis_frequency_at_min_abs_imag_hz": float(freq[min_imag_idx]),
        "eis_z_real_span_ohm": float(np.nanmax(z_real) - np.nanmin(z_real)),
        "eis_low_frequency_nyquist_slope": _low_frequency_nyquist_slope(z_real, z_imag, freq),
    }
    features.update(describe_array("eis_z_abs_ohm", z_abs))
    features.update(describe_array("eis_z_real_ohm", z_real))
    features.update(describe_array("eis_z_imag_ohm", z_imag))
    return features


def _low_frequency_nyquist_slope(z_real: np.ndarray, z_imag: np.ndarray, freq: np.ndarray) -> float:
    """Descriptive low-frequency Nyquist slope; not a fitted Warburg parameter."""
    if len(freq) < 4:
        return float("nan")
    order = np.argsort(freq); low = order[: max(4, len(order) // 4)]
    x = z_real[low]; y = -z_imag[low]; valid = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[valid], y[valid], 1)[0]) if valid.sum() >= 3 else float("nan")
