import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.core import paths
from app.backend.main import app
from app.backend.services.run_service import (
    create_live_run,
    load_live_run,
    update_live_run,
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

    path = (
        tmp_path
        / name
    )

    nib.save(
        nib.Nifti1Image(
            values,
            np.eye(
                4,
                dtype=np.float64,
            ),
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


def _create_completed_run(
    tmp_path: Path,
    live_runs: Path,
) -> str:
    case_id = _create_case(
        tmp_path
    )

    run = create_live_run(
        case_id
    )

    artifact_root = (
        live_runs
        / run.run_id
        / "artifacts"
    )

    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_bytes = (
        json.dumps(
            {
                "status": "synthetic",
            },
            sort_keys=True,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    (
        artifact_root
        / "result.json"
    ).write_bytes(
        result_bytes
    )

    (
        artifact_root
        / "final_mask.nii.gz"
    ).write_bytes(
        b"synthetic-nifti"
    )

    np.save(
        artifact_root
        / "final_mask.npy",
        np.zeros(
            (2, 2, 2),
            dtype=np.uint8,
        ),
        allow_pickle=False,
    )

    update_live_run(
        run.run_id,
        status="complete",
        stage="complete",
        progress=1.0,
        artifacts={
            "result_json": (
                "artifacts/result.json"
            ),
            "final_mask_nifti": (
                "artifacts/final_mask.nii.gz"
            ),
            "final_mask_npy": (
                "artifacts/final_mask.npy"
            ),
        },
    )

    return run.run_id


def test_completed_run_exposes_names_not_paths(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    _, live_runs = isolated_runtime

    run_id = _create_completed_run(
        tmp_path,
        live_runs,
    )

    response = client.get(
        f"/api/v1/runs/{run_id}"
    )

    assert response.status_code == 200

    payload = response.json()

    assert "artifacts" not in payload

    assert payload[
        "available_artifacts"
    ] == [
        "final_mask_nifti",
        "final_mask_npy",
        "result_json",
    ]

    serialized = json.dumps(
        payload
    )

    assert (
        "artifacts/final_mask"
        not in serialized
    )


def test_result_json_artifact_delivery(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    _, live_runs = isolated_runtime

    run_id = _create_completed_run(
        tmp_path,
        live_runs,
    )

    response = client.get(
        f"/api/v1/runs/{run_id}/"
        "artifacts/result_json"
    )

    assert response.status_code == 200

    assert (
        response.headers[
            "content-type"
        ].split(";")[0]
        == "application/json"
    )

    assert response.json() == {
        "status": "synthetic",
    }


def test_nifti_artifact_delivery(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    _, live_runs = isolated_runtime

    run_id = _create_completed_run(
        tmp_path,
        live_runs,
    )

    response = client.get(
        f"/api/v1/runs/{run_id}/"
        "artifacts/final_mask_nifti"
    )

    assert response.status_code == 200

    assert (
        response.headers[
            "content-type"
        ].split(";")[0]
        == "application/gzip"
    )

    assert response.content == (
        b"synthetic-nifti"
    )


def test_npy_artifact_delivery(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    _, live_runs = isolated_runtime

    run_id = _create_completed_run(
        tmp_path,
        live_runs,
    )

    response = client.get(
        f"/api/v1/runs/{run_id}/"
        "artifacts/final_mask_npy"
    )

    assert response.status_code == 200

    assert (
        response.headers[
            "content-type"
        ].split(";")[0]
        == "application/octet-stream"
    )

    assert response.content.startswith(
        b"\x93NUMPY"
    )


def test_artifact_rejected_before_completion(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    case_id = _create_case(
        tmp_path
    )

    run = create_live_run(
        case_id
    )

    response = client.get(
        f"/api/v1/runs/{run.run_id}/"
        "artifacts/result_json"
    )

    assert response.status_code == 409


def test_unknown_artifact_returns_404(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    _, live_runs = isolated_runtime

    run_id = _create_completed_run(
        tmp_path,
        live_runs,
    )

    response = client.get(
        f"/api/v1/runs/{run_id}/"
        "artifacts/not_an_artifact"
    )

    assert response.status_code == 404


def test_artifact_malformed_run_id_returns_400(
    isolated_runtime: tuple[Path, Path],
) -> None:
    response = client.get(
        "/api/v1/runs/not-a-run-id/"
        "artifacts/result_json"
    )

    assert response.status_code == 400


def test_artifact_unknown_run_returns_404(
    isolated_runtime: tuple[Path, Path],
) -> None:
    response = client.get(
        "/api/v1/runs/"
        "run_00000000000000000000000000000000/"
        "artifacts/result_json"
    )

    assert response.status_code == 404


def test_persisted_path_tampering_is_rejected(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    _, live_runs = isolated_runtime

    run_id = _create_completed_run(
        tmp_path,
        live_runs,
    )

    run = load_live_run(
        run_id
    )

    update_live_run(
        run_id,
        artifacts={
            **run.artifacts,
            "result_json": (
                "../outside.json"
            ),
        },
    )

    response = client.get(
        f"/api/v1/runs/{run_id}/"
        "artifacts/result_json"
    )

    assert response.status_code == 500


def test_artifact_symlink_escape_is_rejected(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    _, live_runs = isolated_runtime

    run_id = _create_completed_run(
        tmp_path,
        live_runs,
    )

    run_root = (
        live_runs
        / run_id
    )

    artifact_path = (
        run_root
        / "artifacts"
        / "result.json"
    )

    artifact_path.unlink()

    outside = (
        tmp_path
        / "outside.json"
    )

    outside.write_text(
        '{"outside": true}\n',
        encoding="utf-8",
    )

    artifact_path.symlink_to(
        outside
    )

    response = client.get(
        f"/api/v1/runs/{run_id}/"
        "artifacts/result_json"
    )

    assert response.status_code == 500


def test_artifact_directory_symlink_escape_is_rejected(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
) -> None:
    _, live_runs = isolated_runtime

    run_id = _create_completed_run(
        tmp_path,
        live_runs,
    )

    run_root = (
        live_runs
        / run_id
    )

    artifact_root = (
        run_root
        / "artifacts"
    )

    outside = (
        tmp_path
        / "outside_artifacts"
    )

    outside.mkdir()

    (
        outside
        / "result.json"
    ).write_text(
        '{"outside": true}\n',
        encoding="utf-8",
    )

    for path in artifact_root.iterdir():
        path.unlink()

    artifact_root.rmdir()

    artifact_root.symlink_to(
        outside,
        target_is_directory=True,
    )

    response = client.get(
        f"/api/v1/runs/{run_id}/"
        "artifacts/result_json"
    )

    assert response.status_code == 500
