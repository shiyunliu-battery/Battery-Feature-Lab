"""Compile portable, spectrum-scoped EIS evidence and validation artifacts."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from battery_feature_lab.analysis.schema import AnalysisResult
from battery_feature_lab.analysis.summary import compile_summary
from battery_feature_lab.analysis.writer import (
    canonical_sha256,
    sha256_file,
    validate_payload,
    write_json,
)
from battery_feature_lab.bds_provider import bds_distribution
from battery_feature_lab.eis.analysis import analyze_spectrum
from battery_feature_lab.eis.comparison import compare_spectra
from battery_feature_lab.eis.intake import read_spectra
from battery_feature_lab.eis.schema import (
    EIS_EVIDENCE_SCHEMA,
    EIS_METADATA_SCHEMA,
    EIS_SUMMARY_SCHEMA,
    EIS_VALIDATION_SCHEMA,
    EISConfig,
)
from battery_feature_lab.eis.summary import compact_eis_summary


def compile_eis(source: Path, config: EISConfig) -> AnalysisResult:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(config.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("EIS output directory must be empty; use a new directory for each run")
    spectra, normalized, intake = read_spectra(source, config)
    versions = {}
    for package in ("battery-feature-lab", bds_distribution()[0], "eisfit", "impedance", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    versions["bfl_eis_source_sha256"] = canonical_sha256(
        {path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    )
    source_hash = sha256_file(source)
    run_id = (
        "bfl-eis-"
        + canonical_sha256(
            {
                "source_sha256": source_hash,
                "configuration": config.configuration(),
                "software_versions": versions,
            }
        )[:16]
    )
    calls, records, refs = [], [], {}
    for spectrum in spectra:
        ref = {
            "source_file": source.name,
            "source_sha256": source_hash,
            "sheet": intake["reader_metadata"].get("sheet_name", config.sheet),
            "source_rows": sorted(spectrum["source_rows"]),
            "row_index_basis": "zero_based_BDS_parsed_table",
        }
        refs[spectrum["spectrum_uid"]] = ref
        records.extend(analyze_spectrum(spectrum, ref, config, calls))
    records.extend(compare_spectra(spectra, records, refs, config))
    evidence = {
        "schema_version": "bfl.eis-analysis/0.1",
        "modality": "eis",
        "run_id": run_id,
        "configuration": config.configuration(),
        "records": records,
    }
    identities = {s["conditions"]["cell_id"]["value"] for s in spectra}
    cell_id = (
        next(iter(identities))
        if len(identities) == 1 and None not in identities
        else "multiple_or_unknown"
    )
    summary = compile_summary(records, run_id=run_id, cell_id=cell_id)
    summary = compact_eis_summary(summary)
    metadata = {
        "schema_version": "bfl.eis-metadata/0.1",
        "source": {
            "filename": source.name,
            "sha256": source_hash,
            "metadata_artifact": config.metadata_artifact,
            "reader_metadata": intake["reader_metadata"],
        },
        "spectra": [
            {k: s[k] for k in ("spectrum_uid", "spectrum_id", "conditions")} for s in spectra
        ],
    }
    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate EIS record IDs")
    if any(
        item["evidence"]["record_id"] not in ids
        for dimension in summary["dimensions"].values()
        for item in dimension
    ):
        raise ValueError("Unresolved compact EIS evidence reference")
    if any(
        reference not in ids
        for r in records
        for reference in r["attributes"].get("supporting_record_ids", [])
    ):
        raise ValueError("Unresolved EIS comparison evidence reference")
    schemas = {}
    for name, payload, schema in (
        ("analysis_evidence", evidence, EIS_EVIDENCE_SCHEMA),
        ("analysis_results", summary, EIS_SUMMARY_SCHEMA),
        ("analysis_metadata", metadata, EIS_METADATA_SCHEMA),
    ):
        errors = validate_payload(payload, schema)
        if errors:
            raise ValueError(f"Invalid {name}: {errors}")
        schemas[name] = {"valid": True, "errors": []}
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        output / name
        for name in (
            "normalized_eis.parquet",
            "eis_conversion_report.json",
            "analysis_metadata.json",
            "analysis_results.json",
            "analysis_evidence.json",
            "analysis_validation.json",
        )
    ]
    normalized.write_parquet(paths[0])
    for path, payload in zip(paths[1:5], (intake, metadata, summary, evidence), strict=True):
        write_json(path, payload)
    excluded = sum(len(s["excluded_rows"]) for s in intake["spectra"])
    validation = {
        "schema_version": "bfl.eis-validation/0.1",
        "run_id": run_id,
        "status": "warning"
        if excluded or any(r["quality"]["status"] != "ok" for r in records)
        else "ok",
        "artifacts": {p.name: {"filename": p.name, "sha256": sha256_file(p)} for p in paths[:5]},
        "schema_validation": schemas,
        "row_accounting": {
            "input_rows": intake["input_rows"],
            "retained_rows": normalized.height,
            "excluded_rows": excluded,
            "accounted": intake["input_rows"] == normalized.height + excluded,
        },
        "provider_calls": calls,
        "software_versions": versions,
    }
    errors = validate_payload(validation, EIS_VALIDATION_SCHEMA)
    if errors:
        raise ValueError(f"Invalid EIS validation: {errors}")
    validation["schema_validation"]["analysis_validation"] = {"valid": True, "errors": []}
    write_json(paths[5], validation)
    return AnalysisResult(
        output_dir=output,
        normalized_data_path=paths[0],
        input_report_path=paths[1],
        analysis_metadata_path=paths[2],
        analysis_results_path=paths[3],
        analysis_evidence_path=paths[4],
        analysis_validation_path=paths[5],
        records=tuple(records),
    )
