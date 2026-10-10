"""Regression tests for case-bound scientific PLY reconstruction."""

from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.services import viewer_mesh_service as meshes


@pytest.fixture
def mesh_case(tmp_path, monkeypatch):
    case_id = "case_" + "a" * 32
    run_id = "run_" + "b" * 32

    source_path = tmp_path / "t1n.nii.gz"
    mask_path = tmp_path / "final_mask.nii.gz"

    affine = np.eye(4)
    affine[0, 0] = -1
    affine[0, 3] = 7

    source = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros_like(source)

    mask[2:4, 3:5, 1:3] = 1

    nib.save(nib.Nifti1Image(source, affine), source_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)

    state = {
        "run": SimpleNamespace(
            case_id=case_id,
            status="complete",
        )
    }

    monkeypatch.setattr(
        meshes, "load_live_run",
        lambda _run_id: state["run"],
    )

    monkeypatch.setattr(
        meshes, "_modality_path",
        lambda _case_id, _modality: source_path,
    )

    monkeypatch.setattr(
        meshes, "resolve_live_artifact",
        lambda _run_id, _name: SimpleNamespace(path=mask_path),
    )

    return case_id, run_id, mask_path, affine, state


def test_mesh_preserves_physical_ras_coordinates(mesh_case):
    case_id, run_id, _, _, _ = mesh_case

    payload = meshes.viewer_mesh_ply(case_id, run_id, "final")

    header_end = payload.index(b"end_header\n") + len(b"end_header\n")
    header = payload[:header_end].decode("ascii")

    assert header.startswith("ply\n")
    assert "format binary_little_endian 1.0" in header
    assert "physical RAS+" in header

    def count(element):
        return int(
            next(
                line.split()[-1]
                for line in header.splitlines()
                if line.startswith(f"element {element} ")
            )
        )

    vertex_count = count("vertex")
    face_count = count("face")

    assert vertex_count > 0
    assert face_count > 0

    vertices = np.frombuffer(
        payload,
        dtype="<f4",
        count=vertex_count * 3,
        offset=header_end,
    ).reshape(-1, 3)

    assert np.allclose(
        vertices.min(axis=0),
        [3.5, 2.5, 0.5],
        atol=1e-5,
    )
    assert np.allclose(
        vertices.max(axis=0),
        [5.5, 4.5, 2.5],
        atol=1e-5,
    )

    faces = np.frombuffer(
        payload,
        dtype=np.dtype([
            ("count", "u1"),
            ("indices", "<i4", (3,)),
        ]),
        count=face_count,
        offset=header_end + vertex_count * 12,
    )

    assert np.all(faces["count"] == 3)
    assert np.all(faces["indices"] >= 0)
    assert np.all(faces["indices"] < vertex_count)


def test_mesh_api_rejects_incomplete_or_wrong_case(mesh_case):
    case_id, run_id, _, _, state = mesh_case

    route = (
        f"/api/v1/cases/{case_id}/viewer/"
        f"meshes/{run_id}/final"
    )

    with TestClient(app) as client:
        response = client.get(route)

        assert response.status_code == 200
        assert response.content.startswith(b"ply\n")

        state["run"].status = "running"
        assert client.get(route).status_code == 409

        state["run"].status = "complete"

        wrong_case = (
            "/api/v1/cases/case_" + "c" * 32
            + f"/viewer/meshes/{run_id}/final"
        )
        assert client.get(wrong_case).status_code == 409

        invalid_layer = (
            f"/api/v1/cases/{case_id}/viewer/"
            f"meshes/{run_id}/unknown"
        )
        assert client.get(invalid_layer).status_code == 400


def test_mesh_rejects_misaligned_mask(mesh_case):
    case_id, run_id, mask_path, affine, _ = mesh_case

    shifted = affine.copy()
    shifted[1, 3] += 4.0

    mask = np.zeros((8, 8, 8), dtype=np.float32)
    mask[2:4, 3:5, 1:3] = 1

    nib.save(nib.Nifti1Image(mask, shifted), mask_path)

    with pytest.raises(meshes.ViewerMeshGeometryError):
        meshes.viewer_mesh_ply(case_id, run_id, "final")


def test_mesh_rejects_empty_segmentation(mesh_case):
    case_id, run_id, mask_path, affine, _ = mesh_case

    empty = np.zeros((8, 8, 8), dtype=np.float32)
    nib.save(nib.Nifti1Image(empty, affine), mask_path)

    with pytest.raises(
        meshes.ViewerMeshGeometryError,
        match="no surface",
    ):
        meshes.viewer_mesh_ply(case_id, run_id, "final")
