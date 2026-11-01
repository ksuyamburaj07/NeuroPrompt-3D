"""Coronal and sagittal frozen MRI HTTP regression tests."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.backend.main import app
from app.backend.services import (
    frozen_multiplane_service as service,
)
from app.backend.services.frozen_slice_service import (
    FrozenSliceError,
)


client = TestClient(app)

BASE = "/api/v1/frozen-test/cases"


def url(case_index, plane):
    return f"{BASE}/{case_index}/slices/{plane}"


@pytest.mark.parametrize(
    "plane,index,expected_size",
    [
        ("coronal", 54, (182, 182)),
        ("sagittal", 105, (218, 182)),
    ],
)
def test_real_authenticated_png(plane, index, expected_size):
    response = client.get(
        url(27, plane),
        params={
            "index": index,
            "modality": "t2f",
            "layer": "removed",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"

    with Image.open(BytesIO(response.content)) as image:
        image.verify()

    with Image.open(BytesIO(response.content)) as image:
        assert image.size == expected_size


@pytest.mark.parametrize(
    "plane,index",
    [
        ("coronal", -1),
        ("coronal", 218),
        ("sagittal", -1),
        ("sagittal", 182),
    ],
)
def test_out_of_bounds_index_rejected(plane, index):
    response = client.get(
        url(27, plane),
        params={"index": index},
    )

    assert response.status_code == 422


@pytest.mark.parametrize("plane", ["coronal", "sagittal"])
def test_invalid_modality_rejected(plane):
    response = client.get(
        url(27, plane),
        params={
            "index": 35,
            "modality": "invalid",
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("plane", ["coronal", "sagittal"])
def test_invalid_layer_rejected(plane):
    response = client.get(
        url(27, plane),
        params={
            "index": 35,
            "layer": "ground_truth",
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("plane", ["coronal", "sagittal"])
def test_missing_index_rejected(plane):
    response = client.get(url(27, plane))

    assert response.status_code == 422


@pytest.mark.parametrize("plane", ["coronal", "sagittal"])
def test_post_not_allowed(plane):
    response = client.post(
        url(27, plane),
        params={"index": 35},
    )

    assert response.status_code == 405


@pytest.mark.parametrize("plane", ["coronal", "sagittal"])
def test_invalid_case_index_rejected(plane):
    response = client.get(
        url(0, plane),
        params={"index": 35},
    )

    assert response.status_code == 422


@pytest.mark.parametrize("plane", ["coronal", "sagittal"])
def test_evidence_error_is_sanitized(plane, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FrozenSliceError(
            "SECRET_INTERNAL_EVIDENCE_PATH"
        )

    monkeypatch.setattr(
        service,
        "render_frozen_nonaxial_png",
        unavailable,
    )

    response = client.get(
        url(27, plane),
        params={"index": 35},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Verified frozen MRI slice unavailable."
    )
    assert "SECRET_INTERNAL" not in response.text


def test_unregistered_plane_not_available():
    response = client.get(
        url(27, "oblique"),
        params={"index": 35},
    )

    assert response.status_code == 404
