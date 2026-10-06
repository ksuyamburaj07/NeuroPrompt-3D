from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.core import paths
from app.backend.main import app


client = TestClient(app)


@pytest.fixture
def isolated_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    live_cases = tmp_path / "live_cases"
    live_runs = tmp_path / "live_runs"

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

    return live_cases


def _nifti_bytes(
    tmp_path: Path,
    name: str,
    *,
    shape: tuple[int, int, int] = (6, 5, 4),
    affine: np.ndarray | None = None,
    dtype: np.dtype = np.dtype(np.float32),
) -> bytes:
    if affine is None:
        affine = np.diag(
            [1.25, 1.5, 2.0, 1.0]
        )

    values = np.zeros(
        shape,
        dtype=dtype,
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


def _valid_files(
    tmp_path: Path,
) -> dict[str, tuple[str, bytes, str]]:
    return {
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


def test_validate_stage_get_and_delete_case(
    tmp_path: Path,
    isolated_runtime: Path,
) -> None:
    response = client.post(
        "/api/v1/cases/validate",
        files=_valid_files(tmp_path),
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["valid"] is True
    assert payload["status"] == "ready"
    assert payload["ready_for_inference"] is True

    assert payload["geometry"]["shape_xyz"] == [
        6,
        5,
        4,
    ]

    assert payload["geometry"]["tensor_shape_dhw"] == [
        4,
        5,
        6,
    ]

    assert payload["ground_truth"]["provided"] is False

    assert payload["modalities"]["t1n"]["research_core_name"] == "T1"
    assert payload["modalities"]["t1c"]["research_core_name"] == "T1ce"
    assert payload["modalities"]["t2w"]["research_core_name"] == "T2"
    assert payload["modalities"]["t2f"]["research_core_name"] == "FLAIR"

    case_id = payload["case_id"]

    assert case_id is not None

    case_root = isolated_runtime / case_id

    assert case_root.is_dir()
    assert (case_root / "case.json").is_file()

    get_response = client.get(
        f"/api/v1/cases/{case_id}"
    )

    assert get_response.status_code == 200
    assert get_response.json()["case_id"] == case_id

    delete_response = client.delete(
        f"/api/v1/cases/{case_id}"
    )

    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "case_id": case_id,
        "deleted": True,
    }

    assert not case_root.exists()


def test_validation_rejects_mri_shape_mismatch(
    tmp_path: Path,
    isolated_runtime: Path,
) -> None:
    files = _valid_files(tmp_path)

    files["t2f"] = (
        "t2f.nii.gz",
        _nifti_bytes(
            tmp_path,
            "t2f_bad_shape.nii.gz",
            shape=(6, 5, 3),
        ),
        "application/gzip",
    )

    response = client.post(
        "/api/v1/cases/validate",
        files=files,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["valid"] is False
    assert payload["ready_for_inference"] is False
    assert payload["case_id"] is None

    assert payload["errors"]
    assert payload["errors"][0]["code"] == "nifti_validation_failed"
    assert payload["errors"][0]["field"] == "t2f"

    if isolated_runtime.exists():
        assert list(isolated_runtime.iterdir()) == []


def test_validation_rejects_affine_mismatch(
    tmp_path: Path,
    isolated_runtime: Path,
) -> None:
    files = _valid_files(tmp_path)

    bad_affine = np.diag(
        [1.25, 1.5, 2.0, 1.0]
    )

    bad_affine[0, 3] = 5.0

    files["t1c"] = (
        "t1c.nii.gz",
        _nifti_bytes(
            tmp_path,
            "t1c_bad_affine.nii.gz",
            affine=bad_affine,
        ),
        "application/gzip",
    )

    response = client.post(
        "/api/v1/cases/validate",
        files=files,
    )

    payload = response.json()

    assert payload["valid"] is False
    assert payload["errors"][0]["field"] == "t1c"

    if isolated_runtime.exists():
        assert list(isolated_runtime.iterdir()) == []


def test_optional_ground_truth_is_validated(
    tmp_path: Path,
    isolated_runtime: Path,
) -> None:
    files = _valid_files(tmp_path)

    segmentation = np.zeros(
        (6, 5, 4),
        dtype=np.uint8,
    )

    segmentation[2:4, 2:4, 1:3] = 3

    segmentation_path = (
        tmp_path
        / "segmentation.nii.gz"
    )

    nib.save(
        nib.Nifti1Image(
            segmentation,
            np.diag(
                [1.25, 1.5, 2.0, 1.0]
            ),
        ),
        segmentation_path,
    )

    files["segmentation"] = (
        "segmentation.nii.gz",
        segmentation_path.read_bytes(),
        "application/gzip",
    )

    response = client.post(
        "/api/v1/cases/validate",
        files=files,
    )

    payload = response.json()

    assert payload["valid"] is True
    assert payload["ground_truth"]["provided"] is True
    assert payload["ground_truth"]["valid"] is True
    assert payload["ground_truth"]["tensor_shape_dhw"] == [
        4,
        5,
        6,
    ]
    assert payload["ground_truth"]["labels"] == [
        0,
        3,
    ]

    case_id = payload["case_id"]

    client.delete(
        f"/api/v1/cases/{case_id}"
    )


def test_validation_rejects_ground_truth_shape_mismatch(
    tmp_path: Path,
    isolated_runtime: Path,
) -> None:
    files = _valid_files(tmp_path)

    files["segmentation"] = (
        "segmentation.nii.gz",
        _nifti_bytes(
            tmp_path,
            "segmentation_bad_shape.nii.gz",
            shape=(6, 5, 3),
            dtype=np.dtype(np.uint8),
        ),
        "application/gzip",
    )

    response = client.post(
        "/api/v1/cases/validate",
        files=files,
    )

    payload = response.json()

    assert payload["valid"] is False
    assert payload["case_id"] is None
    assert payload["ground_truth"]["valid"] is False
    assert (
        payload["errors"][0]["code"]
        == "ground_truth_validation_failed"
    )

    if isolated_runtime.exists():
        assert list(isolated_runtime.iterdir()) == []


def test_validation_reports_missing_modalities(
    isolated_runtime: Path,
) -> None:
    response = client.post(
        "/api/v1/cases/validate"
    )

    payload = response.json()

    assert payload["valid"] is False
    assert payload["ready_for_inference"] is False

    fields = {
        issue["field"]
        for issue in payload["errors"]
    }

    assert fields == {
        "t1n",
        "t1c",
        "t2w",
        "t2f",
    }

    if isolated_runtime.exists():
        assert list(isolated_runtime.iterdir()) == []
