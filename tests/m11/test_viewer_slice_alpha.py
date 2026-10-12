"""M11E5C: 3D display-only MRI foreground transparency."""

from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.backend.api import cases as cases_api
from app.backend.main import app
from app.backend.services import viewer_service as viewer


@pytest.mark.parametrize(
    "plane,index",
    [
        ("axial", 0),
        ("coronal", 1),
        ("sagittal", 0),
    ],
)
def test_transparent_slice_keeps_grayscale_and_orientation(
    monkeypatch, plane, index,
):
    volume = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)

    foreground = np.zeros_like(volume, dtype=bool)
    foreground[0, 0, 0] = True
    foreground[1, 1, 0] = True
    foreground[2, 2, 1] = True

    monkeypatch.setattr(
        viewer,
        "_case_volume",
        lambda case_id, modality: (volume, None, None),
    )
    monkeypatch.setattr(
        viewer,
        "_case_source_foreground",
        lambda case_id, modality: foreground,
    )

    original = Image.open(BytesIO(
        viewer.viewer_slice_png("case", "t1n", plane, index)
    ))
    transparent = Image.open(BytesIO(
        viewer.viewer_slice_png(
            "case", "t1n", plane, index,
            transparent_background=True,
        )
    ))

    assert original.mode == "L"
    assert transparent.mode == "RGBA"

    gray = np.asarray(original)
    rgba = np.asarray(transparent)

    for channel in range(3):
        np.testing.assert_array_equal(rgba[:, :, channel], gray)

    if plane == "axial":
        expected = foreground[:, :, index].T[::-1, :]
    elif plane == "coronal":
        expected = foreground[:, index, :].T[::-1, :]
    else:
        expected = foreground[index, :, :].T[::-1, :]

    np.testing.assert_array_equal(
        rgba[:, :, 3],
        expected.astype(np.uint8) * 255,
    )

    # Raw grayscale 0 may be valid foreground; it must stay opaque.
    if plane == "axial" and index == 0:
        assert rgba[-1, 0, 0] == 0
        assert rgba[-1, 0, 3] == 255


def test_slice_api_defaults_to_original_and_accepts_alpha(
    monkeypatch,
):
    received = []

    def fake_slice(
        case_id, modality, plane, index,
        transparent_background=False,
    ):
        received.append(transparent_background)
        return b"test-png"

    monkeypatch.setattr(cases_api, "viewer_slice_png", fake_slice)

    path = "/api/v1/cases/sample/viewer/slices/t1n/axial/0"

    with TestClient(app) as client:
        original = client.get(path)
        transparent = client.get(
            path + "?transparent_background=true"
        )
        invalid = client.get(
            path + "?transparent_background=invalid"
        )

    assert original.status_code == 200
    assert transparent.status_code == 200
    assert invalid.status_code == 422
    assert received == [False, True]
