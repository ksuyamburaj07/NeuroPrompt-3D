"""Real-app integration tests for restricted run discovery."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.api import research_run_discovery as discovery


URL = "/api/v1/research/runs"

RUN_ID = (
    "run_16a49dbf55664daca7c26fc0cb722124"
)


def _client(peer="127.0.0.1"):
    return TestClient(
        app,
        base_url="http://127.0.0.1:8000",
        client=(peer, 50000),
    )


def _fake_record():
    return SimpleNamespace(
        run_id=RUN_ID,
        status="complete",
        stage="complete",
        progress=1.0,
        updated_at=datetime(
            2026, 10, 6, tzinfo=timezone.utc
        ),
        case_id="private-case",
        available_artifacts=["private-artifact.json"],
    )


def test_discovery_is_registered_in_openapi():
    schema = app.openapi()

    operation = (
        schema.get("paths", {})
        .get(URL, {})
        .get("get")
    )

    assert operation is not None
    assert "research-discovery" in operation.get("tags", [])


def test_discovery_disabled_in_real_app(monkeypatch):
    monkeypatch.delenv(
        "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY",
        raising=False,
    )

    def forbidden_listing(*, limit):
        raise AssertionError(
            "Disabled route must not scan storage."
        )

    monkeypatch.setattr(
        discovery,
        "list_recent_live_runs",
        forbidden_listing,
    )

    response = _client().get(URL)

    assert response.status_code == 404
    assert "no-store" in response.headers["cache-control"]


def test_local_real_app_response_is_minimal(monkeypatch):
    monkeypatch.setenv(
        "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY",
        "1",
    )

    observed = []

    def fake_listing(*, limit):
        observed.append(limit)
        return [_fake_record()]

    monkeypatch.setattr(
        discovery,
        "list_recent_live_runs",
        fake_listing,
    )

    response = _client().get(
        URL,
        params={"limit": 10},
        headers={"Origin": "http://127.0.0.1:5173"},
    )

    assert response.status_code == 200
    assert observed == [10]

    payload = response.json()

    assert payload["count"] == 1
    assert payload["limit"] == 10

    assert set(payload["runs"][0]) == {
        "run_id",
        "status",
        "stage",
        "progress",
        "updated_at",
    }

    assert payload["runs"][0]["run_id"] == RUN_ID
    assert "private-case" not in response.text
    assert "private-artifact.json" not in response.text

    assert "no-store" in response.headers["cache-control"]
    assert response.headers[
        "x-content-type-options"
    ] == "nosniff"


def test_remote_client_cannot_use_proxy_headers(monkeypatch):
    monkeypatch.setenv(
        "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY",
        "1",
    )

    def forbidden_listing(*, limit):
        raise AssertionError(
            "Remote client must be denied before scan."
        )

    monkeypatch.setattr(
        discovery,
        "list_recent_live_runs",
        forbidden_listing,
    )

    response = _client(
        peer="198.51.100.10",
    ).get(
        URL,
        headers={
            "Origin": "http://127.0.0.1:5173",
            "X-Forwarded-For": "127.0.0.1",
        },
    )

    assert response.status_code == 404


@pytest.mark.parametrize("limit", [0, 51, "bad"])
def test_invalid_limits_in_real_app(monkeypatch, limit):
    monkeypatch.setenv(
        "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY",
        "1",
    )

    response = _client().get(
        URL,
        params={"limit": limit},
    )

    assert response.status_code == 422


def test_storage_error_is_sanitized(monkeypatch):
    monkeypatch.setenv(
        "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY",
        "1",
    )

    def broken_listing(*, limit):
        raise RuntimeError("/private/path")

    monkeypatch.setattr(
        discovery,
        "list_recent_live_runs",
        broken_listing,
    )

    response = _client().get(URL)

    assert response.status_code == 503
    assert "/private/path" not in response.text
