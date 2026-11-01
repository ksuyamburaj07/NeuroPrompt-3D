"""HTTP contract tests for the frozen axial evidence endpoint."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.backend.main import app
from app.backend.api import frozen_test as router
from app.backend.services import frozen_slice_service as service


client = TestClient(app)
BASE = "/api/v1/frozen-test/cases"


def sample_png():
    buffer = BytesIO()
    Image.new("RGB", (182, 218)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def test_authenticated_png_response(monkeypatch):
    calls = []
    png = sample_png()

    def fake_render(index, z, *, modality, layer):
        calls.append((index, z, modality, layer))
        return png

    monkeypatch.setattr(
        router,
        "render_frozen_axial_png",
        fake_render,
    )

    response = client.get(
        f"{BASE}/27/slices/axial",
        params={
            "z": 35,
            "modality": "t2f",
            "layer": "removed",
        },
    )

    assert response.status_code == 200
    assert response.content == png
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert calls == [(27, 35, "t2f", "removed")]


def test_default_options(monkeypatch):
    calls = []

    def fake_render(index, z, *, modality, layer):
        calls.append((index, z, modality, layer))
        return sample_png()

    monkeypatch.setattr(
        router,
        "render_frozen_axial_png",
        fake_render,
    )

    response = client.get(
        f"{BASE}/1/slices/axial",
        params={"z": 111},
    )

    assert response.status_code == 200
    assert calls == [(1, 111, "t2f", "removed")]


@pytest.mark.parametrize(
    "index,params",
    [
        ("0", {"z": 35}),
        ("130", {"z": 35}),
        ("abc", {"z": 35}),
        ("27", {}),
        ("27", {"z": -1}),
        ("27", {"z": 182}),
        ("27", {"z": "abc"}),
        ("27", {"z": 35, "modality": "invalid"}),
        ("27", {"z": 35, "layer": "invalid"}),
    ],
)
def test_invalid_requests_return_422(index, params):
    response = client.get(
        f"{BASE}/{index}/slices/axial",
        params=params,
    )
    assert response.status_code == 422


def test_missing_evidence_returns_503(monkeypatch):
    def unavailable(*args, **kwargs):
        raise service.FrozenSliceError(
            "Sensitive internal path: /data/private/source"
        )

    monkeypatch.setattr(
        router,
        "render_frozen_axial_png",
        unavailable,
    )

    response = client.get(
        f"{BASE}/27/slices/axial",
        params={"z": 35},
    )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert "/data/private/source" not in response.text
    assert "Sensitive internal path" not in response.text


def test_tampered_prediction_manifest_returns_503(
    tmp_path, monkeypatch
):
    altered = tmp_path / "altered-manifest.json"
    altered.write_bytes(
        service.PREDICTION_MANIFEST.read_bytes() + b"\n"
    )

    monkeypatch.setattr(
        service,
        "PREDICTION_MANIFEST",
        altered,
    )

    response = client.get(
        f"{BASE}/27/slices/axial",
        params={"z": 35},
    )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert str(tmp_path) not in response.text


def test_post_not_permitted():
    response = client.post(
        f"{BASE}/27/slices/axial",
        params={"z": 35},
    )

    assert response.status_code == 405
