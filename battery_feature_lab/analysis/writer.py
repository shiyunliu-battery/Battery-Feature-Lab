"""Strict JSON writing, validation, and artifact hashing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def artifact_reference(path: str | Path) -> str:
    """Return a portable, host-independent reference for one file artifact.

    Public JSON binds source artifacts to a SHA-256 digest elsewhere in the
    contract, so the filename is sufficient for identification without
    exposing a workstation directory tree.
    """

    return Path(path).name


def validate_payload(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Return all Draft 2020-12 validation errors in stable path order."""

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    schema_errors = [
        f"{'/'.join(str(part) for part in error.path) or '$'}: {error.message}" for error in errors
    ]
    return sorted([*schema_errors, *_non_finite_errors(payload)])


def _non_finite_errors(value: Any, path: str = "$") -> list[str]:
    """Return JSON paths for NaN and infinity, which JSON Schema treats as numbers."""

    if isinstance(value, float) and not math.isfinite(value):
        return [f"{path}: non-finite number {value!r} is not valid JSON"]
    if isinstance(value, dict):
        return [
            error
            for key, item in value.items()
            for error in _non_finite_errors(item, f"{path}/{key}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            error
            for index, item in enumerate(value)
            for error in _non_finite_errors(item, f"{path}/{index}")
        ]
    return []


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write finite, UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.write(b"\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    """Hash canonical compact JSON for an explicitly documented content scope."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
