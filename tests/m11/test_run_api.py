from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.backend.api.runs as runs_api

from app.backend.core import paths
from app.backend.schemas.runs import RunRecord
from app.backend.main import app
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
