"""Synthetic tests for gated cortical-contour PNG rendering."""

from io import BytesIO
import struct

import numpy as np
import pytest
from PIL import Image

from app.backend.services import viewer_cortex_contour_service as service
from app.backend.services.viewer_cortex_service import CortexUnavailable


def fake_ply(x_offset=0):
    vertices = np.array([
        [1 + x_offset, 2, 3],
        [4 + x_offset, 2, 5],
        [1 + x_offset, 5, 5],
    ], dtype="<f4")

    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"comment coordinates physical_RAS_plus_mm\n"
        b"element vertex 3\n"
        b"property float x\n"
        b"property float y\n"
        b"property float z\n"
        b"element face 1\n"
        b"property list uchar int vertex_indices\n"
        b"end_header\n"
    )

    return (
        header
        + vertices.tobytes()
        + struct.pack("<Biii", 3, 0, 1, 2)
    )


@pytest.fixture
def eligible_case(monkeypatch):
    meshes = {
        "lh": fake_ply(0),
        "rh": fake_ply(6),
    }

    monkeypatch.setattr(
        service,
        "viewer_cortex_mesh_pair",
        lambda case_id: meshes,
    )

    monkeypatch.setattr(
        service,
        "viewer_metadata",
        lambda case_id: {
            "shape_ras_xyz": [12, 12, 12],
            "affine_ras": np.eye(4).tolist(),
        },
    )

    return meshes


def test_original_ply_parser():
    vertices, faces = service._parse_original_pial_ply(fake_ply())

    assert vertices.shape == (3, 3)
    assert faces.shape == (1, 3)

    np.testing.assert_array_equal(faces, [[0, 1, 2]])


@pytest.mark.parametrize(
    ("plane", "index"),
    [
        ("axial", 4),
        ("coronal", 3),
        ("sagittal", 2),
    ],
)
def test_transparent_contour_image(eligible_case, plane, index):
    payload = service.viewer_cortex_contours_png(
        "synthetic_case",
        plane,
        index,
    )

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")

    image = Image.open(BytesIO(payload))

    assert image.mode == "RGBA"
    assert image.size == (12, 12)

    pixels = np.asarray(image)
    assert np.any(pixels[:, :, 3] > 0)
    assert np.any(pixels[:, :, 3] == 0)


def test_both_hemisphere_colors(eligible_case):
    payload = service.viewer_cortex_contours_png(
        "synthetic_case",
        "axial",
        4,
    )

    pixels = np.asarray(Image.open(BytesIO(payload)))
    visible = pixels[pixels[:, :, 3] > 150]

    assert len(visible) > 0

    # LH gold has red greater than blue.
    assert np.any(visible[:, 0] > visible[:, 2])

    # RH cyan has blue greater than red.
    assert np.any(visible[:, 2] > visible[:, 0])


def test_invalid_plane_rejected(eligible_case):
    with pytest.raises(ValueError):
        service.viewer_cortex_contours_png(
            "synthetic_case", "diagonal", 4
        )


def test_out_of_range_slice_rejected(eligible_case):
    with pytest.raises(ValueError):
        service.viewer_cortex_contours_png(
            "synthetic_case", "axial", 99
        )


def test_corrupt_ply_rejected():
    with pytest.raises(ValueError):
        service._parse_original_pial_ply(b"not a ply")


def test_provisional_cortex_stays_unavailable(monkeypatch):
    def denied(case_id):
        raise CortexUnavailable(
            "Cortical reconstruction is unavailable."
        )

    monkeypatch.setattr(
        service,
        "viewer_cortex_mesh_pair",
        denied,
    )

    with pytest.raises(CortexUnavailable):
        service.viewer_cortex_contours_png(
            "provisional_case", "axial", 4
        )
