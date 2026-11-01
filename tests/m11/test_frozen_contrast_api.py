"""HTTP compatibility and volume contrast regression tests."""

from hashlib import sha256
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.backend.main import app
from app.backend.services import (
    frozen_slice_service as axial,
    frozen_multiplane_service as nonaxial,
)
from app.backend.services.frozen_slice_service import (
    FrozenSliceError,
)


client = TestClient(app)

CASES = [
    (
        "axial",
        {"z": 35},
        (163, 105),
        "06d50726df01f41bb7e3daa0d1fa034"
        "ef753fd45eab3dff2a3a639c584046bde",
    ),
    (
        "coronal",
        {"index": 54},
        (146, 105),
        "e4164d5fede22dc5fae41392bb28f37"
        "af03c62feb4edd346cd341f46f9c369c5",
    ),
    (
        "sagittal",
        {"index": 105},
        (146, 54),
        "ccefee18364d0676d41ab0bc0729adc"
        "bee271128e99f93eb52cfc4107d2578d0",
    ),
]


def request(plane, coordinate, contrast=None):
    params = {
        **coordinate,
        "modality": "t2f",
        "layer": "none",
    }

    if contrast is not None:
        params["contrast"] = contrast

    return client.get(
        f"/api/v1/frozen-test/cases/27/slices/{plane}",
        params=params,
    )


@pytest.mark.parametrize(
    "plane,coordinate,pixel,digest",
    CASES,
)
def test_default_preserves_legacy_png(
    plane, coordinate, pixel, digest
):
    original = request(plane, coordinate)
    explicit = request(plane, coordinate, "slice")

    assert original.status_code == 200
    assert explicit.status_code == 200
    assert original.content == explicit.content
    assert sha256(original.content).hexdigest() == digest

    assert original.headers["content-type"] == "image/png"
    assert original.headers["cache-control"] == "no-store"
    assert (
        original.headers["x-content-type-options"]
        == "nosniff"
    )


@pytest.mark.parametrize(
    "plane,coordinate,pixel,digest",
    CASES,
)
def test_volume_contrast_real_voxel(
    plane, coordinate, pixel, digest
):
    response = request(
        plane, coordinate, "volume"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"

    row, column = pixel

    with Image.open(BytesIO(response.content)) as image:
        image.verify()

    with Image.open(BytesIO(response.content)) as image:
        rgb = image.convert("RGB").getpixel(
            (column, row)
        )

    assert rgb == (207, 207, 207)


@pytest.mark.parametrize(
    "plane,coordinate,pixel,digest",
    CASES,
)
def test_invalid_contrast_returns_422(
    plane, coordinate, pixel, digest
):
    response = request(
        plane, coordinate, "not-a-valid-mode"
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "plane,coordinate,pixel,digest",
    CASES,
)
def test_volume_contrast_failure_is_sanitized(
    plane, coordinate, pixel, digest, monkeypatch
):
    def unavailable(*args, **kwargs):
        raise FrozenSliceError(
            "SECRET_INTERNAL_MRI_SOURCE_PATH"
        )

    monkeypatch.setattr(
        axial,
        "_frozen_mri_volume_contrast_window",
        unavailable,
    )

    monkeypatch.setattr(
        nonaxial,
        "_frozen_mri_volume_contrast_window",
        unavailable,
    )

    response = request(
        plane, coordinate, "volume"
    )

    assert response.status_code == 503
    assert "SECRET_INTERNAL" not in response.text
    assert "SECRET_INTERNAL" not in str(response.json())
