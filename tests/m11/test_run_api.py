from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.backend.api.runs as runs_api

from app.backend.core import paths
from app.backend.schemas.runs import RunRecord
from app.backend.main import app
from app.backend.services.run_service import (
    update_live_run,
)
from src.pipeline.policy import (
    FROZEN_VARIANCE_THRESHOLD,
)


client = TestClient(app)


@pytest.fixture
def isolated_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    live_cases = (
        tmp_path
        / "live_cases"
    )

    live_runs = (
        tmp_path
        / "live_runs"
    )

    monkeypatch.setattr(
        paths,
        "LIVE_CASES_ROOT",
        live_cases,
    )

    monkeypatch.setattr(
        paths,
        "LIVE_RUNS_ROOT",
        live_runs,
    )

    return (
        live_cases,
        live_runs,
    )


def _nifti_bytes(
    tmp_path: Path,
    name: str,
) -> bytes:
    values = np.zeros(
        (6, 5, 4),
        dtype=np.float32,
    )

    affine = np.diag(
        [1.25, 1.5, 2.0, 1.0]
    )

    path = tmp_path / name

    nib.save(
        nib.Nifti1Image(
            values,
            affine,
        ),
        path,
    )

    return path.read_bytes()


def _create_case(
    tmp_path: Path,
) -> str:
    files = {
        field: (
            f"{field}.nii.gz",
            _nifti_bytes(
                tmp_path,
                f"{field}.nii.gz",
            ),
            "application/gzip",
        )
        for field in (
            "t1n",
            "t1c",
            "t2w",
            "t2f",
        )
    }

    response = client.post(
        "/api/v1/cases/validate",
        files=files,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["valid"] is True

    return payload["case_id"]


def test_create_and_get_queued_run(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    _, live_runs = isolated_runtime

    case_id = _create_case(
        tmp_path
    )

    response = client.post(
        f"/api/v1/cases/{case_id}/runs"
    )

    assert response.status_code == 201

    payload = response.json()

    assert payload["case_id"] == case_id
    assert payload["status"] == "queued"
    assert payload["stage"] == "queued"
    assert payload["progress"] == 0.0

    assert (
        payload["frozen_variance_threshold"]
        == FROZEN_VARIANCE_THRESHOLD
    )

    assert payload["action"] is None
    assert payload["sam_used"] is None
    assert payload["error"] is None

    assert payload[
        "available_artifacts"
    ] == []

    assert "artifacts" not in payload

    run_id = payload["run_id"]

    run_root = (
        live_runs
        / run_id
    )

    assert run_root.is_dir()

    assert (
        run_root
        / "run.json"
    ).is_file()

    get_response = client.get(
        f"/api/v1/runs/{run_id}"
    )

    assert get_response.status_code == 200

    assert (
        get_response.json()
        == payload
    )


def test_create_run_rejects_unknown_case(
    isolated_runtime: tuple[Path, Path],
) -> None:
    response = client.post(
        "/api/v1/cases/"
        "case_00000000000000000000000000000000/"
        "runs"
    )

    assert response.status_code == 404


def test_get_run_rejects_malformed_identifier(
    isolated_runtime: tuple[Path, Path],
) -> None:
    response = client.get(
        "/api/v1/runs/not-a-run-id"
    )

    assert response.status_code == 400


def test_get_run_returns_404_for_unknown_run(
    isolated_runtime: tuple[Path, Path],
) -> None:
    response = client.get(
        "/api/v1/runs/"
        "run_00000000000000000000000000000000"
    )

    assert response.status_code == 404


def test_execute_endpoint_launches_queued_run(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = _create_case(
        tmp_path
    )

    create_response = client.post(
        f"/api/v1/cases/{case_id}/runs"
    )

    assert create_response.status_code == 201

    original = RunRecord.model_validate(
        create_response.json()
    )

    def fake_launch(
        run_id: str,
    ) -> RunRecord:
        assert run_id == original.run_id

        return original.model_copy(
            update={
                "status": "running",
                "stage": "validating",
                "progress": 0.01,
                "worker_pid": 12345,
            }
        )

    monkeypatch.setattr(
        runs_api,
        "launch_run_worker",
        fake_launch,
    )

    response = client.post(
        f"/api/v1/runs/{original.run_id}/execute"
    )

    assert response.status_code == 202

    payload = response.json()

    assert payload["run_id"] == original.run_id
    assert payload["status"] == "running"
    assert payload["stage"] == "validating"
    assert payload["progress"] == 0.01
    assert payload["worker_pid"] == 12345


def test_delete_case_rejects_active_run(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    case_id = _create_case(
        tmp_path
    )

    create_response = client.post(
        f"/api/v1/cases/{case_id}/runs"
    )

    assert create_response.status_code == 201

    run_id = create_response.json()[
        "run_id"
    ]

    delete_response = client.delete(
        f"/api/v1/cases/{case_id}"
    )

    assert delete_response.status_code == 409

    assert (
        client.get(
            f"/api/v1/cases/{case_id}"
        ).status_code
        == 200
    )

    assert (
        client.get(
            f"/api/v1/runs/{run_id}"
        ).status_code
        == 200
    )


def test_delete_case_after_completed_run_keeps_run(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    case_id = _create_case(
        tmp_path
    )

    create_response = client.post(
        f"/api/v1/cases/{case_id}/runs"
    )

    assert create_response.status_code == 201

    run_id = create_response.json()[
        "run_id"
    ]

    update_live_run(
        run_id,
        status="complete",
        stage="complete",
        progress=1.0,
    )

    delete_response = client.delete(
        f"/api/v1/cases/{case_id}"
    )

    assert delete_response.status_code == 200

    assert (
        client.get(
            f"/api/v1/cases/{case_id}"
        ).status_code
        == 404
    )

    # Completed result metadata is a separate live resource.
    run_response = client.get(
        f"/api/v1/runs/{run_id}"
    )

    assert run_response.status_code == 200
    assert (
        run_response.json()["status"]
        == "complete"
    )


def test_concurrent_launch_requests_start_exactly_one_worker(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two simultaneous requests cannot spawn two workers."""

    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock
    from time import sleep

    from app.backend.services import worker_service

    case_id = _create_case(tmp_path)

    response = client.post(
        f"/api/v1/cases/{case_id}/runs"
    )
    assert response.status_code == 201

    run_id = response.json()["run_id"]

    first_claim_reached = Event()
    release_first_claim = Event()
    counter_lock = Lock()
    spawned = [0]

    original_update = worker_service.update_live_run

    def delayed_update(*args, **kwargs):
        if kwargs.get("status") == "running":
            first_claim_reached.set()

            if not release_first_claim.wait(timeout=5):
                raise RuntimeError(
                    "Timed out waiting to release test claim"
                )

        return original_update(*args, **kwargs)

    def fake_popen(*args, **kwargs):
        with counter_lock:
            spawned[0] += 1
        return object()

    monkeypatch.setattr(
        worker_service,
        "update_live_run",
        delayed_update,
    )

    monkeypatch.setattr(
        worker_service.subprocess,
        "Popen",
        fake_popen,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            worker_service.launch_run_worker,
            run_id,
        )

        assert first_claim_reached.wait(timeout=5)

        second = pool.submit(
            worker_service.launch_run_worker,
            run_id,
        )

        try:
            sleep(0.1)
        finally:
            release_first_claim.set()

        first_result = first.result(timeout=5)

        with pytest.raises(
            worker_service.RunLaunchConflict
        ):
            second.result(timeout=5)

    assert first_result.run_id == run_id
    assert first_result.status == "running"
    assert spawned[0] == 1

    persisted = worker_service.load_live_run(
        run_id
    )
    assert persisted.status == "running"


def test_idempotency_key_replays_same_run(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    from uuid import uuid4

    case_id = _create_case(tmp_path)
    endpoint = f"/api/v1/cases/{case_id}/runs"
    headers = {"Idempotency-Key": str(uuid4())}

    first = client.post(endpoint, headers=headers)
    second = client.post(endpoint, headers=headers)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["run_id"] == second.json()["run_id"]
    assert second.json()["status"] == "queued"


def test_distinct_idempotency_keys_create_distinct_runs(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    from uuid import uuid4

    case_id = _create_case(tmp_path)
    endpoint = f"/api/v1/cases/{case_id}/runs"

    first = client.post(
        endpoint,
        headers={"Idempotency-Key": str(uuid4())},
    )
    second = client.post(
        endpoint,
        headers={"Idempotency-Key": str(uuid4())},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["run_id"] != second.json()["run_id"]


def test_invalid_idempotency_key_is_rejected(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    case_id = _create_case(tmp_path)

    response = client.post(
        f"/api/v1/cases/{case_id}/runs",
        headers={"Idempotency-Key": "not-a-uuid"},
    )

    assert response.status_code == 400


def test_legacy_run_creation_without_key_remains_supported(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    case_id = _create_case(tmp_path)
    endpoint = f"/api/v1/cases/{case_id}/runs"

    first = client.post(endpoint)
    second = client.post(endpoint)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["run_id"] != second.json()["run_id"]


def test_parallel_idempotent_creation_reuses_one_run(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from uuid import uuid4
    from app.backend.services.run_service import create_live_run

    case_id = _create_case(tmp_path)
    key = str(uuid4())

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                create_live_run,
                case_id,
                idempotency_key=key,
            )
            for _ in range(4)
        ]
        results = [future.result(timeout=10) for future in futures]

    assert len({record.run_id for record in results}) == 1
    assert all(record.status == "queued" for record in results)
