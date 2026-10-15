"""Exercise installed BDS through both public BFL paths, including provenance."""

import json

import polars as pl

import bfl
from battery_feature_lab.bds_provider import bds_distribution


def test_installed_bds_cycling_and_eis_provenance(tmp_path):
    provider, provider_version = bds_distribution()
    cycling = tmp_path / "cycling.csv"
    pl.DataFrame({
        "test_time_s": [0., 1., 2., 3.],
        "voltage_v": [3., 3.1, 3.2, 3.3],
        "current_a": [1., 1., 1., 1.],
    }).write_csv(cycling)
    result = bfl.analyze(cycling, output_dir=tmp_path / "cycling-out", input_adapter="bds")
    metadata = json.loads(result.analysis_metadata_path.read_text())
    assert provider in json.dumps(metadata)
    validation = json.loads(result.analysis_validation_path.read_text())
    assert validation["software_versions"][provider] == provider_version
    assert (tmp_path / "cycling-out" / "bds_conversion_report.json").is_file()

    eis = tmp_path / "eis.csv"
    pl.DataFrame({
        "Frequency_Hz": [1000., 100., 10., 1.],
        "Real_Ohm": [0.01, 0.02, 0.03, 0.04],
        "Imag_Ohm": [-0.01, -0.02, -0.02, -0.01],
    }).write_csv(eis)
    result = bfl.analyze_eis(eis, tmp_path / "eis-out", single_spectrum=True,
                             run_kk=False, run_fit=False)
    report = json.loads((tmp_path / "eis-out" / "eis_conversion_report.json").read_text())
    assert report["provider"] == provider
    assert report["provider_version"] == provider_version
    assert report["retained_rows"] == 4
    validation = json.loads(result.analysis_validation_path.read_text())
    assert validation["software_versions"][provider] == provider_version
