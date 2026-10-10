"""Tests for read-only MRI-derived anatomical context."""

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.services import viewer_anatomy_service as anatomy
from app.backend.services.viewer_mesh_service import (
    ViewerMeshGeometryError,
)


@pytest.fixture
def anatomy_case(tmp_path, monkeypatch):
    case_id = "case_" + "a" * 32
    path = tmp_path / "t1n.nii.gz"

    affine = np.diag([-2.0, 1.0, 1.5, 1.0])
    affine[:3, 3] = [50.0, -20.0, 10.0]

    image = np.zeros((16, 16, 16), dtype=np.float32)
    image[3:10, 4:12, 2:9] = 100

    nib.save(nib.Nifti1Image(image, affine), path)

    def resolve(requested_case, modality):
        if requested_case != case_id or modality != "t1n":
            raise FileNotFoundError("Unknown staged MRI.")
        return path

    monkeypatch.setattr(anatomy, "_modality_path", resolve)

    return case_id, path, affine


def parse_vertices(payload):
    header_end = (
        payload.index(b"end_header\n") +
        len(b"end_header\n")
    )

    header = payload[:header_end].decode("ascii")

    vertex_count = int(
        next(
            line.split()[-1]
            for line in header.splitlines()
            if line.startswith("element vertex ")
        )
    )

    triangle_count = int(
        next(
            line.split()[-1]
            for line in header.splitlines()
            if line.startswith("element face ")
        )
    )

    assert vertex_count > 0
    assert triangle_count > 0

    assert len(payload) == (
        header_end +
        12 * vertex_count +
        13 * triangle_count
    )

    return np.frombuffer(
        payload,
        dtype="<f4",
        count=vertex_count * 3,
        offset=header_end,
    ).reshape(-1, 3)


def test_anatomy_uses_physical_ras_coordinates(anatomy_case):
    case_id, _, _ = anatomy_case

    payload = anatomy.viewer_anatomy_ply(case_id)
    vertices = parse_vertices(payload)

    assert payload.startswith(b"ply\n")
    assert np.isfinite(vertices).all()

    # Known synthetic native voxel region:
    # X 3..9, Y 4..11, Z 2..8
    # with negative X affine, nonunit spacing and translation.
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)

    assert 28 <= minimum[0] <= 36
    assert 42 <= maximum[0] <= 48

    assert -19 <= minimum[1] <= -13
    assert -12 <= maximum[1] <= -6

    assert 10 <= minimum[2] <= 16
    assert 19 <= maximum[2] <= 26


@pytest.mark.parametrize("bad_data", ["empty", "nonfinite"])
def test_anatomy_rejects_invalid_foreground(
    anatomy_case,
    bad_data,
):
    case_id, path, affine = anatomy_case

    values = np.zeros((16, 16, 16), dtype=np.float32)

    if bad_data == "nonfinite":
        values[5, 5, 5] = np.nan

    nib.save(nib.Nifti1Image(values, affine), path)

    with pytest.raises(ViewerMeshGeometryError):
        anatomy.viewer_anatomy_ply(case_id)


def test_anatomy_http_contract(anatomy_case):
    case_id, _, _ = anatomy_case

    url = f"/api/v1/cases/{case_id}/viewer/anatomy/brain"

    with TestClient(app) as client:
        response = client.get(url)

        assert response.status_code == 200
        assert response.content.startswith(b"ply\n")
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-mesh-coordinates"] == (
            "RAS+ millimetres"
        )
        assert response.headers["x-mesh-step-voxels"] == "2"

        parse_vertices(response.content)

        missing = client.get(
            "/api/v1/cases/case_" +
            "b" * 32 +
            "/viewer/anatomy/brain"
        )
        assert missing.status_code == 404



def test_anatomy_detail_options(anatomy_case):
    case_id, _, _ = anatomy_case
    url = f"/api/v1/cases/{case_id}/viewer/anatomy/brain"

    with TestClient(app) as client:
        balanced = client.get(url)
        detailed = client.get(url + "?step=1")
        invalid = client.get(url + "?step=3")

    assert balanced.status_code == 200
    assert detailed.status_code == 200
    assert invalid.status_code == 400

    assert balanced.headers["x-mesh-step-voxels"] == "2"
    assert detailed.headers["x-mesh-step-voxels"] == "1"

    assert len(detailed.content) > len(balanced.content)
    parse_vertices(detailed.content)



def test_anatomy_surface_treatment(anatomy_case):
    case_id, _, _ = anatomy_case

    url = f"/api/v1/cases/{case_id}/viewer/anatomy/brain"

    with TestClient(app) as client:
        raw = client.get(url + "?step=1&finish=raw")
        default = client.get(url + "?step=1")
        soft = client.get(url + "?step=1&finish=soft")
        invalid = client.get(url + "?step=1&finish=unknown")
        unsupported = client.get(url + "?step=2&finish=soft")

    assert raw.status_code == 200
    assert default.status_code == 200
    assert soft.status_code == 200

    assert invalid.status_code == 400
    assert unsupported.status_code == 400

    assert raw.content == default.content
    assert raw.content != soft.content

    assert raw.headers["x-mesh-finish"] == "raw"
    assert soft.headers["x-mesh-finish"] == "soft"

    raw_vertices = parse_vertices(raw.content)
    soft_vertices = parse_vertices(soft.content)

    assert np.isfinite(raw_vertices).all()
    assert np.isfinite(soft_vertices).all()
