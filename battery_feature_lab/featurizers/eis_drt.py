"""EIS and DRT feature extraction interfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_feature_lab.featurizers.base import BaseFeaturizer
from battery_feature_lab.featurizers.common import describe_array
from battery_feature_lab.schemas import FeatureTable


class EISDRTFeaturizer(BaseFeaturizer):
    """Extract simple EIS features when impedance columns are available.

    DRT peak extraction requires a dedicated inversion method. This class provides a stable
    schema and useful first-order impedance descriptors while leaving DRT-specific peak
    areas as NaN until DRT inputs are supplied by a future adapter.
    """

    name = "eis_drt_features"

    def extract(self, frame: pd.DataFrame) -> FeatureTable:
        required = {"frequency_hz", "z_real_ohm", "z_imag_ohm"}
        if not required.issubset(frame.columns):
            return FeatureTable(self.name, pd.DataFrame())

        rows: list[dict[str, float | int | str]] = []
        group_cols = ["cell_id"]
        if "cycle_index" in frame.columns:
            group_cols.append("cycle_index")
        for key, group in frame.groupby(group_cols, sort=True):
            features = _eis_features(group)
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
    features: dict[str, float] = {
        "eis_point_count": int(len(data)),
        "eis_high_freq_intercept_ohm": float(z_real[high_freq_idx]),
        "eis_min_abs_imag_z_real_ohm": float(z_real[min_imag_idx]),
        "eis_min_abs_imag_frequency_hz": float(freq[min_imag_idx]),
        "eis_semicircle_width_ohm": float(np.nanmax(z_real) - np.nanmin(z_real)),
        "eis_warburg_slope": _warburg_slope(z_real, z_imag, freq),
        "drt_peak_area_diffusion": float("nan"),
        "drt_peak_area_charge_transfer": float("nan"),
        "drt_peak_area_sei": float("nan"),
        "drt_peak_area_ohmic": float("nan"),
    }
    features.update(describe_array("eis_z_abs_ohm", z_abs))
    features.update(describe_array("eis_z_real_ohm", z_real))
    features.update(describe_array("eis_z_imag_ohm", z_imag))
    return features


def _warburg_slope(z_real: np.ndarray, z_imag: np.ndarray, freq: np.ndarray) -> float:
    if len(freq) < 4:
        return float("nan")
    order = np.argsort(freq)
    low = order[: max(4, len(order) // 4)]
    x = z_real[low]
    y = -z_imag[low]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    return float(np.polyfit(x[valid], y[valid], 1)[0])
