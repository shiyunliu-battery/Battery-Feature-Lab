from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from battery_feature_lab.cli import main
from battery_feature_lab.analysis.degradation_tags import build_degradation_tags, mann_kendall_sen_slope
from battery_feature_lab.bds_adapter.readers import read_bds_export
from battery_feature_lab.pipeline import FeaturePipeline, PipelineConfig
from battery_feature_lab.schemas import DiagnosticConfig, ExportConfig, FeatureConfig, ReaderConfig


def make_synthetic_bds(cycles: int = 110) -> pd.DataFrame:
    rows = []
    nominal = 1.1
    t = 0.0
    for cycle in range(1, cycles + 1):
        fade = 1.0 - 0.0008 * cycle
        qmax = nominal * fade
        step = 0

        charge_q = np.linspace(0, qmax, 40)
        charge_v = 3.0 + 0.55 * (charge_q / qmax) + 0.015 * np.sin(charge_q / qmax * np.pi)
        for q, v in zip(charge_q, charge_v):
            rows.append(
                {
                    "cell": "synthetic_cell",
                    "cycle": cycle,
                    "step": step,
                    "time": t,
                    "voltage": v,
                    "current": 1.1,
                    "temperature": 25 + 0.01 * cycle,
                    "charge_capacity": q,
                    "discharge_capacity": 0.0,
                }
            )
            t += 30

        step += 1
        rest_v = np.linspace(charge_v[-1], charge_v[-1] - 0.04, 12)
        for v in rest_v:
            rows.append(
                {
                    "cell": "synthetic_cell",
                    "cycle": cycle,
                    "step": step,
                    "time": t,
                    "voltage": v,
                    "current": 0.0,
                    "temperature": 25 + 0.01 * cycle,
                    "charge_capacity": qmax,
                    "discharge_capacity": 0.0,
                }
            )
            t += 20

        step += 1
        discharge_q = np.linspace(0, qmax * 0.995, 45)
        discharge_v = 3.45 - 0.6 * (discharge_q / qmax) - 0.01 * cycle / cycles
        for q, v in zip(discharge_q, discharge_v):
            rows.append(
                {
                    "cell": "synthetic_cell",
                    "cycle": cycle,
                    "step": step,
                    "time": t,
                    "voltage": v,
                    "current": -1.1,
                    "temperature": 25 + 0.01 * cycle,
                    "charge_capacity": qmax,
                    "discharge_capacity": q,
                }
            )
            t += 30
    return pd.DataFrame(rows)


def test_pipeline_extracts_core_tables(tmp_path: Path) -> None:
    path = tmp_path / "bds.csv"
    make_synthetic_bds().to_csv(path, index=False)

    pipeline = FeaturePipeline(
        PipelineConfig(
            reader=ReaderConfig(cell_id="synthetic_cell"),
            features=FeatureConfig(nominal_capacity_ah=1.1),
            export=ExportConfig(output_dir=tmp_path / "out"),
        )
    )
    tables = pipeline.run(path)

    assert not tables["cycle_features"].empty
    assert not tables["delta_q_features"].empty
    assert not tables["ica_dva_features"].empty
    assert not tables["relaxation_features"].empty
    assert not tables["stress_features"].empty
    assert (tmp_path / "out" / "llm_context.jsonl").exists()


def test_cli_writes_outputs(tmp_path: Path) -> None:
    path = tmp_path / "bds.csv"
    make_synthetic_bds(cycles=15).to_csv(path, index=False)

    exit_code = main(
        [
            "extract",
            str(path),
            "--output-dir",
            str(tmp_path / "cli_out"),
            "--cell-id",
            "synthetic_cell",
            "--nominal-capacity-ah",
            "1.1",
            "--reference-cycle",
            "2",
            "--target-cycle",
            "10",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "cli_out" / "cycle_features.parquet").exists()
    assert (tmp_path / "cli_out" / "run_metadata.json").exists()


def test_reader_handles_bds_total_time_dchg_and_generic_capacity(tmp_path: Path) -> None:
    path = tmp_path / "realistic_bds.csv"
    raw = pd.DataFrame(
        {
            "DataPoint": range(1, 12),
            "Step Type": [
                "Rest",
                "Rest",
                "CC Chg",
                "CC Chg",
                "Rest",
                "CC DChg",
                "CC DChg",
                "Rest",
                "CC Chg",
                "CC DChg",
                "Rest",
            ],
            "Time": [
                "00:00:00",
                "00:00:01",
                "00:00:00",
                "00:00:01",
                "00:00:00",
                "00:00:00",
                "00:00:01",
                "00:00:00",
                "00:00:00",
                "00:00:00",
                "00:00:00",
            ],
            "Total Time": [
                "00:00:00",
                "00:00:01",
                "00:00:02",
                "00:00:03",
                "00:00:04",
                "00:00:05",
                "00:00:06",
                "00:00:07",
                "00:00:08",
                "00:00:09",
                "00:00:10",
            ],
            "Current(A)": [0.0, 0.0, 0.5, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0],
            "Voltage(V)": [3.2, 3.2, 3.3, 3.4, 3.4, 3.3, 3.1, 3.1, 3.4, 3.0, 3.0],
            "Capacity(Ah)": [0.0, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.0, 0.1, 0.1, 0.0],
            "Energy(Wh)": [0.0] * 11,
        }
    )
    raw.to_csv(path, index=False)

    normalized = read_bds_export(path, ReaderConfig(cell_id="realistic_cell"))

    assert normalized["time_s"].notna().all()
    assert normalized["time_s"].iloc[-1] == 10
    assert set(normalized["step_type"]) == {"charge", "discharge", "rest"}
    assert normalized["cycle_index"].nunique() == 2
    assert normalized["charge_capacity_ah"].max() == 0.2
    assert normalized["discharge_capacity_ah"].max() == 0.2
    assert normalized.loc[normalized["step_type"] == "discharge", "current_a"].lt(0).all()


def test_mann_kendall_sen_slope_detects_monotonic_fade() -> None:
    x = np.arange(20)
    y = 1.0 - 0.01 * x

    trend = mann_kendall_sen_slope(x, y)

    assert trend["p_value"] < 0.05
    assert trend["sen_slope"] < 0


def _stress_row(cell_id: str, high_soc_rest: float, c_rate_variance: float) -> dict:
    return {
        "cell_id": cell_id,
        "high_soc_rest_fraction": high_soc_rest,
        "c_rate_variance": c_rate_variance,
        "max_instant_discharge_c_rate": 1.0,
    }


def test_single_cell_high_soc_rest_fires_via_absolute_threshold() -> None:
    # A single cell has no population, so the batch criterion cannot apply; the absolute
    # (config/domain-knowledge) threshold must still let the tag fire for deployment.
    high = pd.DataFrame([_stress_row("A", high_soc_rest=0.6, c_rate_variance=0.01)])
    tags = build_degradation_tags(stress_features=high, config=DiagnosticConfig())
    assert "high_soc_rest_exposure" in set(tags["signal"])
    evidence = tags.loc[tags["signal"] == "high_soc_rest_exposure", "evidence"].iloc[0]
    assert "absolute threshold" in evidence


def test_single_cell_low_exposure_does_not_false_fire() -> None:
    low = pd.DataFrame([_stress_row("A", high_soc_rest=0.3, c_rate_variance=0.01)])
    tags = build_degradation_tags(stress_features=low, config=DiagnosticConfig())
    signals = set(tags["signal"]) if "signal" in tags.columns else set()
    assert "high_soc_rest_exposure" not in signals
    assert "dynamic_current_variance" not in signals


def test_batch_outlier_is_labelled_relative() -> None:
    vals = [0.009, 0.010, 0.011, 0.010, 0.009, 0.011, 0.010, 0.010, 0.060]
    batch = pd.DataFrame(
        [_stress_row(f"C{i}", high_soc_rest=0.1, c_rate_variance=v) for i, v in enumerate(vals)]
    )
    tags = build_degradation_tags(stress_features=batch, config=DiagnosticConfig())
    crv = tags[tags["signal"] == "dynamic_current_variance"]
    assert len(crv) == 1
    assert crv["cell_id"].iloc[0] == "C8"
    assert "batch-relative outlier" in crv["evidence"].iloc[0]


def test_robust_mad_outlier_not_masked_by_single_extreme() -> None:
    # Under a mean+k*std rule a single extreme cell inflates std and can mask itself (here
    # mean+2*std ~= 5.47 > 5.0, so the outlier would NOT fire). The robust median+k*MAD rule must
    # still flag it.
    vals = [0.010, 0.012, 0.011, 0.013, 5.0]
    batch = pd.DataFrame(
        [_stress_row(f"C{i}", high_soc_rest=0.1, c_rate_variance=v) for i, v in enumerate(vals)]
    )
    tags = build_degradation_tags(stress_features=batch, config=DiagnosticConfig())
    crv = tags[tags["signal"] == "dynamic_current_variance"]
    assert len(crv) == 1
    assert crv["cell_id"].iloc[0] == "C4"
    assert "MAD" in crv["evidence"].iloc[0]


def test_datasheet_c_rate_tag_requires_spec() -> None:
    fast = pd.DataFrame([_stress_row("A", high_soc_rest=0.1, c_rate_variance=0.01)])
    fast.loc[0, "max_instant_discharge_c_rate"] = 4.5
    no_spec = build_degradation_tags(stress_features=fast, config=DiagnosticConfig())
    with_spec = build_degradation_tags(
        stress_features=fast, config=DiagnosticConfig(datasheet_max_discharge_c_rate=3.0)
    )
    no_signals = set(no_spec["signal"]) if "signal" in no_spec.columns else set()
    assert "high_instantaneous_discharge_rate" not in no_signals
    assert "high_instantaneous_discharge_rate" in set(with_spec["signal"])
