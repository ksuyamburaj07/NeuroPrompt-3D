from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.services import frozen_test_service as service


client = TestClient(app)
URL = "/api/v1/frozen-test/cases"


def test_default_case_page():
    response = client.get(URL)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"

    data = response.json()

    assert data["milestone"] == "M9E6"
    assert data["total_cases"] == 129
    assert data["total_matching"] == 129
    assert len(data["cases"]) == 20
    assert [r["case_index"] for r in data["cases"]] == (
        list(range(1, 21))
    )


def test_final_page():
    response = client.get(
        URL,
        params={"offset": 120, "limit": 20},
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["cases"]) == 9
    assert [r["case_index"] for r in data["cases"]] == (
        list(range(121, 130))
    )


def test_action_filter():
    response = client.get(
        URL,
        params={
            "action": "APPLY_LOCAL_FP",
            "limit": 129,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_matching"] == 44
    assert len(data["cases"]) == 44
    assert all(
        r["action"] == "APPLY_LOCAL_FP"
        for r in data["cases"]
    )


def test_public_records_match_frozen_evidence():
    response = client.get(URL, params={"limit": 1})

    assert response.status_code == 200

    public_case = response.json()["cases"][0]
    frozen_case = service.load_verified_evaluation()[
        "case_results"
    ][0]

    assert public_case["case_index"] == 1
    assert public_case["baseline_dice"] == (
        frozen_case["baseline_dice"]
    )
    assert public_case["final_dice"] == (
        frozen_case["final_dice"]
    )
    assert public_case["hotspot_variance"] == (
        frozen_case["hotspot_variance"]
    )

    assert "case_id" not in public_case
    assert "ground_truth_voxels" not in public_case


def test_invalid_query_parameters():
    invalid = [
        {"offset": -1},
        {"limit": 0},
        {"limit": 130},
        {"action": "UNKNOWN"},
    ]

    for params in invalid:
        response = client.get(URL, params=params)
        assert response.status_code == 422


def test_tampered_evidence_returns_503(
    tmp_path,
    monkeypatch,
):
    altered = tmp_path / "altered.json"

    altered.write_bytes(
        service.EVALUATION_PATH.read_bytes() + b"\n"
    )

    monkeypatch.setattr(
        service,
        "EVALUATION_PATH",
        altered,
    )

    response = client.get(URL)

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Verified frozen test evidence is unavailable."
    )
    assert str(tmp_path) not in response.text


def test_post_is_not_supported():
    response = client.post(URL)

    assert response.status_code == 405
