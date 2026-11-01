"""Tests for read-only saved research-run discovery."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from app.backend.core import paths
from app.backend.services import research_run_listing_service as listing


def _write_run(
    root: Path,
    index: int,
    updated_at: str,
    *,
    stored_id: str | None = None,
) -> str:
    run_id = f"run_{index:032x}"
    directory = root / run_id
    directory.mkdir(parents=True)

    payload = {
        "run_id": stored_id or run_id,
        "case_id": f"case_{index:032x}",
        "status": "complete",
        "stage": "complete",
        "progress": 1.0,
        "created_at": "2026-10-01T10:00:00+00:00",
        "updated_at": updated_at,
        "frozen_variance_threshold": 0.21,
    }

    (directory / "run.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return run_id


@pytest.fixture
def isolated_runs(tmp_path, monkeypatch):
    root = tmp_path / "live_runs"
    monkeypatch.setattr(paths, "LIVE_RUNS_ROOT", root)
    return root


def test_missing_root_returns_empty_without_creating_it(
    isolated_runs,
):
    assert listing.list_recent_live_runs() == []
    assert not isolated_runs.exists()


def test_newest_first_and_bounded_results(isolated_runs):
    first = _write_run(
        isolated_runs, 1, "2026-10-02T10:00:00+00:00"
    )
    second = _write_run(
        isolated_runs, 2, "2026-10-04T10:00:00+00:00"
    )
    third = _write_run(
        isolated_runs, 3, "2026-10-03T10:00:00+00:00"
    )

    results = listing.list_recent_live_runs(limit=2)

    assert [record.run_id for record in results] == [
        second,
        third,
    ]

    assert first not in [record.run_id for record in results]


@pytest.mark.parametrize(
    "limit",
    [0, -1, 51, True, 1.5, "20"],
)
def test_invalid_limit_rejected(isolated_runs, limit):
    with pytest.raises(ValueError):
        listing.list_recent_live_runs(limit=limit)


def test_invalid_and_corrupted_records_excluded(isolated_runs):
    valid = _write_run(
        isolated_runs, 1, "2026-10-04T10:00:00+00:00"
    )

    _write_run(
        isolated_runs,
        2,
        "2026-10-05T10:00:00+00:00",
        stored_id=f"run_{99:032x}",
    )

    malformed_id = isolated_runs / "run_bad"
    malformed_id.mkdir()
    (malformed_id / "run.json").write_text("{}")

    corrupt = isolated_runs / f"run_{3:032x}"
    corrupt.mkdir()
    (corrupt / "run.json").write_text("{invalid json")

    incomplete = isolated_runs / f"run_{4:032x}"
    incomplete.mkdir()

    results = listing.list_recent_live_runs()

    assert [record.run_id for record in results] == [valid]


def test_symlink_entries_are_excluded(
    isolated_runs,
    tmp_path,
):
    valid = _write_run(
        isolated_runs, 1, "2026-10-04T10:00:00+00:00"
    )

    outside = tmp_path / "external_run"
    outside.mkdir()
    (outside / "run.json").write_text("{}")

    (isolated_runs / f"run_{2:032x}").symlink_to(
        outside,
        target_is_directory=True,
    )

    third = isolated_runs / f"run_{3:032x}"
    third.mkdir()

    (third / "run.json").symlink_to(
        isolated_runs / valid / "run.json"
    )

    results = listing.list_recent_live_runs()

    assert [record.run_id for record in results] == [valid]


def test_symlink_storage_root_rejected(
    isolated_runs,
    tmp_path,
):
    actual_root = tmp_path / "actual"
    actual_root.mkdir()

    isolated_runs.symlink_to(
        actual_root,
        target_is_directory=True,
    )

    with pytest.raises(RuntimeError):
        listing.list_recent_live_runs()


def test_scan_limit_fails_explicitly(
    isolated_runs,
    monkeypatch,
):
    for index in range(1, 5):
        _write_run(
            isolated_runs,
            index,
            "2026-10-04T10:00:00+00:00",
        )

    monkeypatch.setattr(
        listing,
        "MAX_SCANNED_ENTRIES",
        3,
    )

    with pytest.raises(RuntimeError):
        listing.list_recent_live_runs()


def test_listing_does_not_modify_saved_metadata(
    isolated_runs,
):
    run_id = _write_run(
        isolated_runs,
        1,
        "2026-10-04T10:00:00+00:00",
    )

    metadata = isolated_runs / run_id / "run.json"
    before = sha256(metadata.read_bytes()).hexdigest()

    assert len(listing.list_recent_live_runs()) == 1

    after = sha256(metadata.read_bytes()).hexdigest()

    assert after == before
