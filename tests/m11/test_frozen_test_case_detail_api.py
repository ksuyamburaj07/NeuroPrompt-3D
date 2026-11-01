from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.services import frozen_test_service as service


client = TestClient(app)
BASE = "/api/v1/frozen-test/cases"


def test_first_case():
    response = client.get(f"{BASE}/1")

    assert response.status_code == 200
    data = response.json()

    assert data["milestone"] == "M9E6"
    assert data["case"]["case_index"] == 1
    assert data["evaluation_sha256"] == service.EXPECTED_SHA256
    assert response.headers["cache-control"] == "no-store"


def test_last_case():
    response = client.get(f"{BASE}/129")

    assert response.status_code == 200
    assert response.json()["case"]["case_index"] == 129


def test_detail_matches_verified_list():
    listing = client.get(
        BASE,
        params={"offset": 63, "limit": 1},
    )

    detail = client.get(f"{BASE}/64")

    assert listing.status_code == 200
    assert detail.status_code == 200

    assert detail.json()["case"] == (
        listing.json()["cases"][0]
    )


def test_raw_case_id_not_exposed():
    response = client.get(f"{BASE}/1")

    assert response.status_code == 200
    assert "case_id" not in response.json()["case"]
    assert "ground_truth_voxels" not in response.json()["case"]


def test_invalid_case_indices_rejected():
    for index in ("0", "130", "-1", "abc"):
        response = client.get(f"{BASE}/{index}")
        assert response.status_code == 422


def test_missing_source_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(
        service,
        "EVALUATION_PATH",
        tmp_path / "missing.json",
    )

    response = client.get(f"{BASE}/1")

    assert response.status_code == 503
    assert str(tmp_path) not in response.text


def test_tampered_source_returns_503(tmp_path, monkeypatch):
    altered = tmp_path / "altered.json"
    altered.write_bytes(
        service.EVALUATION_PATH.read_bytes() + b"\n"
    )

    monkeypatch.setattr(
        service,
        "EVALUATION_PATH",
        altered,
    )

    response = client.get(f"{BASE}/1")

    assert response.status_code == 503


def test_post_is_rejected():
    response = client.post(f"{BASE}/1")

    assert response.status_code == 405
