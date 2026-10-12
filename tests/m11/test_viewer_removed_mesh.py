"""Regression for frozen removal-mask mesh delivery."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.services import viewer_mesh_service as meshes


@pytest.fixture
def removal_case(tmp_path, monkeypatch):
    case_id = "case_" + "a" * 32
    run_id = "run_" + "b" * 32

    affine = np.eye(4)

    source = np.zeros((8, 8, 8), dtype=np.float32)
    baseline = np.zeros_like(source)
    baseline[2:5, 2:5, 2:5] = 1

    final = baseline.copy()
    final[4, 4, 4] = 0

    removed = baseline - final

    paths = {}
    for name, values in {
        "t1n": source,
        "baseline_mask_nifti": baseline,
        "final_mask_nifti": final,
        "removal_mask_nifti": removed,
    }.items():
        path = tmp_path / (name + ".nii.gz")
        nib.save(nib.Nifti1Image(values, affine), path)
        paths[name] = path

    state = {"missing": False}

    monkeypatch.setattr(
        meshes,
        "load_live_run",
        lambda _run_id: SimpleNamespace(
            case_id=case_id,
            status="complete",
        ),
    )

    monkeypatch.setattr(
        meshes,
        "_modality_path",
        lambda _case_id, _modality: paths["t1n"],
    )

    def artifact(_run_id, name):
        if name == "removal_mask_nifti" and state["missing"]:
            raise FileNotFoundError("Removal artifact missing.")

        return SimpleNamespace(path=paths[name])

    monkeypatch.setattr(
        meshes, "resolve_live_artifact", artifact
    )

    return case_id, run_id, paths, state


def test_removed_layer_uses_its_own_frozen_artifact(removal_case):
    case_id, run_id, _, _ = removal_case

    baseline = meshes.viewer_mesh_ply(
        case_id, run_id, "baseline"
    )
    final = meshes.viewer_mesh_ply(
        case_id, run_id, "final"
    )
    removed = meshes.viewer_mesh_ply(
        case_id, run_id, "removed"
    )

    assert removed.startswith(b"ply\n")
    assert removed != baseline
    assert removed != final


def test_missing_removed_artifact_returns_404(removal_case):
    case_id, run_id, _, state = removal_case
    state["missing"] = True

    url = (
        f"/api/v1/cases/{case_id}/viewer/"
        f"meshes/{run_id}/removed"
    )

    with TestClient(app) as client:
        response = client.get(url)

    assert response.status_code == 404


def test_empty_removed_mask_returns_409(removal_case):
    case_id, run_id, paths, _ = removal_case

    empty = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(
        nib.Nifti1Image(empty, np.eye(4)),
        paths["removal_mask_nifti"],
    )

    url = (
        f"/api/v1/cases/{case_id}/viewer/"
        f"meshes/{run_id}/removed"
    )

    with TestClient(app) as client:
        response = client.get(url)

    assert response.status_code == 409
