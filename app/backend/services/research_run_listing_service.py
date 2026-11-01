"""Bounded, read-only discovery of saved development runs.

This service does not create, execute, update, or delete runs.
It must not be exposed publicly without access controls.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from app.backend.core import paths
from app.backend.schemas.runs import RunRecord
from app.backend.services.run_service import load_live_run


RUN_ID_PATTERN = re.compile(r"run_[0-9a-f]{32}")

DEFAULT_LIMIT = 20
MAX_RESULTS = 50
MAX_SCANNED_ENTRIES = 2000


def _valid_timestamp(value: datetime) -> bool:
    """Accept only timestamps with explicit timezone information."""
    return (
        value.tzinfo is not None
        and value.utcoffset() is not None
    )


def list_recent_live_runs(
    *,
    limit: int = DEFAULT_LIMIT,
) -> list[RunRecord]:
    """Return newest verified records from bounded local storage.

    Malformed, incomplete, mismatched, and symlinked entries are
    excluded. Excessive directory inventories fail explicitly
    rather than returning an incorrectly labelled recent list.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
        raise ValueError(
            f"Run listing limit must be between 1 and {MAX_RESULTS}"
        )

    root: Path = paths.LIVE_RUNS_ROOT

    if root.is_symlink():
        raise RuntimeError(
            "Refusing symbolic-link run storage root"
        )

    if not root.exists():
        return []

    if not root.is_dir():
        raise RuntimeError(
            "Run storage root is not a directory"
        )

    verified: list[RunRecord] = []

    for scanned, directory in enumerate(root.iterdir(), start=1):
        if scanned > MAX_SCANNED_ENTRIES:
            raise RuntimeError(
                "Run inventory exceeds the configured scan limit"
            )

        if directory.is_symlink() or not directory.is_dir():
            continue

        run_id = directory.name

        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            continue

        metadata = directory / "run.json"

        if metadata.is_symlink() or not metadata.is_file():
            continue

        try:
            record = load_live_run(run_id)
        except (OSError, ValueError):
            # Do not publish records that failed authentication.
            continue

        if record.run_id != run_id:
            continue

        if not (
            _valid_timestamp(record.created_at)
            and _valid_timestamp(record.updated_at)
        ):
            continue

        verified.append(record)

    verified.sort(
        key=lambda record: (
            record.updated_at.timestamp(),
            record.created_at.timestamp(),
            record.run_id,
        ),
        reverse=True,
    )

    return verified[:limit]
