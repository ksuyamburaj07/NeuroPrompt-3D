"""Persistent live-inference run metadata for the M11 application."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.backend.core import paths
from app.backend.schemas.runs import (
    RunError,
    RunRecord,
    RunStage,
    RunStatus,
)
from app.backend.services.case_service import (
    load_live_case,
)
from src.pipeline.policy import (
    FROZEN_VARIANCE_THRESHOLD,
)


_RUN_ID_PATTERN = re.compile(
    r"^run_[0-9a-f]{32}$"
)


def _utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def _validate_run_id(
    run_id: str,
) -> None:
    if not _RUN_ID_PATTERN.fullmatch(
        run_id
    ):
        raise ValueError(
            "Invalid live run identifier"
        )


def _run_root(
    run_id: str,
) -> Path:
    _validate_run_id(
        run_id
    )

    return (
        paths.LIVE_RUNS_ROOT
        / run_id
    )


def _metadata_path(
    run_id: str,
) -> Path:
    return (
        _run_root(run_id)
        / "run.json"
    )


def _write_run(
    record: RunRecord,
) -> None:
    root = _run_root(
        record.run_id
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        root
        / "run.json"
    )

    temporary = (
        root
        / ".run.json.tmp"
    )

    temporary.write_text(
        json.dumps(
            record.model_dump(
                mode="json"
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        destination,
    )


def _create_live_run_unkeyed(
    case_id: str,
    *,
    explicit_run_id: str | None = None,
) -> RunRecord:
    """Create a queued run for an already validated live case."""

    case = load_live_case(
        case_id
    )

    if not case.ready_for_inference:
        raise ValueError(
            "Live case is not ready for inference"
        )

    paths.ensure_runtime_directories()

    run_id = explicit_run_id or f"run_{uuid4().hex}"

    now = _utc_now()

    record = RunRecord(
        run_id=run_id,
        case_id=case_id,
        status="queued",
        stage="queued",
        progress=0.0,
        created_at=now,
        updated_at=now,
        frozen_variance_threshold=(
            FROZEN_VARIANCE_THRESHOLD
        ),
        action=None,
        gate_state=None,
        hotspot_zyx=None,
        hotspot_variance=None,
        fp_prompt_zyx=None,
        fp_prompt_model_xyz=None,
        sam_used=None,
        sam_refinement_skipped=None,
        semantic_abstention_condition=None,
        artifacts={},
        error=None,
    )

    _write_run(
        record
    )

    return record



def create_live_run(
    case_id: str,
    *,
    idempotency_key: str | None = None,
) -> RunRecord:
    """Create a run, replaying the same identity for the same key."""

    if idempotency_key is None:
        return _create_live_run_unkeyed(case_id)

    import fcntl
    from uuid import UUID, NAMESPACE_URL, uuid5

    try:
        parsed_key = UUID(idempotency_key)
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError(
            "Idempotency-Key must be a canonical UUID v4."
        ) from exc

    if (
        parsed_key.version != 4
        or str(parsed_key) != idempotency_key.lower()
    ):
        raise ValueError(
            "Idempotency-Key must be a canonical UUID v4."
        )

    run_id = "run_" + uuid5(
        NAMESPACE_URL,
        f"neuroprompt3d/live-run/v1/{case_id}/{parsed_key}",
    ).hex

    paths.ensure_runtime_directories()

    lock_path = paths.LIVE_RUNS_ROOT / ".create_run.lock"

    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)

        try:
            try:
                existing = load_live_run(run_id)
            except FileNotFoundError:
                return _create_live_run_unkeyed(
                    case_id,
                    explicit_run_id=run_id,
                )

            if existing.case_id != case_id:
                raise ValueError(
                    "Idempotency key resolved to a different case."
                )

            return existing
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def load_live_run(
    run_id: str,
) -> RunRecord:
    metadata = _metadata_path(
        run_id
    )

    if not metadata.is_file():
        raise FileNotFoundError(
            run_id
        )

    return RunRecord.model_validate_json(
        metadata.read_text(
            encoding="utf-8"
        )
    )


def update_live_run(
    run_id: str,
    *,
    status: RunStatus | None = None,
    stage: RunStage | None = None,
    progress: float | None = None,
    execution_device: str | None = None,
    worker_pid: int | None = None,
    inference_case_id: str | None = None,
    mc_case_seed: int | None = None,
    action: str | None = None,
    gate_state: str | None = None,
    hotspot_zyx: list[int] | None = None,
    hotspot_variance: float | None = None,
    fp_prompt_zyx: list[int] | None = None,
    fp_prompt_model_xyz: list[int] | None = None,
    sam_used: bool | None = None,
    sam_refinement_skipped: bool | None = None,
    semantic_abstention_condition: str | None = None,
    artifacts: dict[str, str] | None = None,
    error: RunError | None = None,
) -> RunRecord:
    """Atomically update one persisted run record."""

    record = load_live_run(
        run_id
    )

    changes: dict[str, object] = {
        "updated_at": _utc_now(),
    }

    optional_updates = {
        "status": status,
        "stage": stage,
        "progress": progress,
        "execution_device": execution_device,
        "worker_pid": worker_pid,
        "inference_case_id": inference_case_id,
        "mc_case_seed": mc_case_seed,
        "action": action,
        "gate_state": gate_state,
        "hotspot_zyx": hotspot_zyx,
        "hotspot_variance": hotspot_variance,
        "fp_prompt_zyx": fp_prompt_zyx,
        "fp_prompt_model_xyz": fp_prompt_model_xyz,
        "sam_used": sam_used,
        "sam_refinement_skipped": sam_refinement_skipped,
        "semantic_abstention_condition": (
            semantic_abstention_condition
        ),
        "artifacts": artifacts,
        "error": error,
    }

    for key, value in (
        optional_updates.items()
    ):
        if value is not None:
            changes[key] = value

    updated = record.model_copy(
        update=changes
    )

    _write_run(
        updated
    )

    return updated


def active_live_run_ids_for_case(
    case_id: str,
) -> list[str]:
    """Return queued/running live runs associated with one case."""

    paths.ensure_runtime_directories()

    active: list[str] = []

    for run_root in sorted(
        paths.LIVE_RUNS_ROOT.iterdir()
    ):
        if (
            not run_root.is_dir()
            or not _RUN_ID_PATTERN.fullmatch(
                run_root.name
            )
        ):
            continue

        metadata = (
            run_root
            / "run.json"
        )

        if not metadata.is_file():
            continue

        record = (
            RunRecord.model_validate_json(
                metadata.read_text(
                    encoding="utf-8"
                )
            )
        )

        if (
            record.case_id == case_id
            and record.status
            in {
                "queued",
                "running",
            }
        ):
            active.append(
                record.run_id
            )

    return active
