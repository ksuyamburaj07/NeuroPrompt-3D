from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.services import frozen_test_service as service


client = TestClient(app)

URL = "/api/v1/frozen-test/summary"


def test_summary_returns_verified_frozen_results():
    response = client.get(URL)

    assert response.status_code == 200

    payload = response.json()

    assert payload["milestone"] == "M9E6"
    assert payload["case_count"] == 129

    assert payload["action_counts"] == {
        "ABSTAIN_BASELINE": 85,
        "APPLY_LOCAL_FP": 44,
    }

    assert (
        payload["overall_summary"]["delta_dice_mean"]
        == -0.003755445490538544
    )

    assert payload["evaluation_sha256"] == (
        service.EXPECTED_SHA256
    )

    assert response.headers["cache-control"] == "no-store"
    assert "case_results" not in payload


def test_missing_frozen_source_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        service,
        "EVALUATION_PATH",
        tmp_path / "missing.json",
    )

    response = client.get(URL)

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Verified frozen test evidence is unavailable."
    )

    assert str(tmp_path) not in response.text


def test_tampered_frozen_source_fails_closed(tmp_path, monkeypatch):
    original = service.EVALUATION_PATH.read_bytes()

    altered = tmp_path / "tampered.json"
    altered.write_bytes(original + b"\n")

    monkeypatch.setattr(
        service,
        "EVALUATION_PATH",
        altered,
    )

    response = client.get(URL)

    assert response.status_code == 503
    assert "case_results" not in response.text


def test_summary_does_not_accept_post():
    response = client.post(URL)

    assert response.status_code == 405
