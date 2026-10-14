"""Capture third-party console output and warnings as portable audit facts."""

from __future__ import annotations

import contextlib
import io
import re
import warnings
from collections import Counter
from collections.abc import Iterator
from typing import Any

_ABSOLUTE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\|/)(?:[^\s\\/:'\"<>|]+[\\/])+[^\s\\/:'\"<>|]*"
)


def portable_message(message: str) -> str:
    """Replace absolute filesystem paths so audit records stay host-independent."""

    return _ABSOLUTE_PATH.sub("<path>", message)


@contextlib.contextmanager
def capture_provider_output() -> Iterator[dict[str, Any]]:
    """Capture stdout, stderr and warnings emitted while a provider runs.

    The yielded dictionary is filled when the block exits, including when the
    provider raises. Console text and warning source locations are discarded
    because they can contain host-local paths; only line counts and warning
    categories and messages are retained.
    """

    report: dict[str, Any] = {}
    stdout, stderr = io.StringIO(), io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                yield report
        finally:
            counts = Counter(
                (warning.category.__name__, portable_message(str(warning.message)))
                for warning in caught
            )
            report.update(
                stdout_line_count=len(stdout.getvalue().splitlines()),
                stderr_line_count=len(stderr.getvalue().splitlines()),
                warnings=[
                    {"category": category, "message": message, "count": count}
                    for (category, message), count in sorted(counts.items())
                ],
            )
