"""Isolated tests for restricted research-run discovery."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.backend.api import research_run_discovery as discovery


URL = "/api/v1/research/runs"
REAL_ID = "run_16a49dbf55664daca7c26fc0cb722124"


def _client(
    *,
    peer="127.0.0.1",
    base_url="http://127.0.0.1:8000",
):
    application = FastAPI()

    application.include_router(
        discovery.router,
        prefix="/api/v1",
    )

    return TestClient(
        application,
        base_url=base_url,
        client=(peer, 50000),
    )


def _saved_record():
    return SimpleNamespace(
        run_id=REAL_ID,
        status="complete",
        stage="complete",
        progress=1.0,
        updated_at=datetime(
            2026, 10, 6, 18, 43,
            tzinfo=timezone.utc,
        ),
        case_id="private-case-identifier",
        mc_case_seed=123,
        available_artifacts=["private.json"],
    )


def test_discovery_disabled_by_default(monkeypatch):
    monkeypatch.delenv(
        "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY",
        raising=False,
    )

    def forbidden_scan(*, limit):
        raise AssertionError(
            "Listing must not run while endpoint is disabled."
        )

    monkeypatch.setattr(
        discovery,
        "list_recent_live_runs",
        forbidden_scan,
    )

    response = _client().get(URL)

    assert response.status_code == 404


def test_local_discovery_returns_minimal_metadata(monkeypatch):
    monkeypatch.setenv(
        "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY",
        "1",
    )

    requested = []

    def fake_list(*, limit):
        requested.append(limit)
        return [_saved_record()]

    monkeypatch.setattr(
        discovery,
        "list_recent_live_runs",
        fake_list,
    )

    response = _client().get(
        URL,
        params={"limit": 10},
        headers={
            "Origin": "http://127.0.0.1:5173",
        },
    )

    assert response.status_code == 200
    assert requested == [10]

    payload = response.json()

    assert payload["count"] == 1
    assert payload["limit"] == 10
    assert len(payload["runs"]) == 1

    item = payload["runs"][0]

    assert set(item) == {
        "run_id",
        "status",
        "stage",
        "progress",
        "updated_at",
    }

    assert item["run_id"] == REAL_ID
    assert item["status"] == "complete"

    raw = response.text

    assert "private-case-identifier" not in raw
    assert "private.json" not in raw
    assert "mc_case_seed" not in raw

    assert "no-store" in response.headers[
        "cache-control"
    ]

    assert response.headers[
        "x-content-type-options"
    ] == "nosniff"


@pytest.mark.parametrize(
    "peer,host,origin",
    [
        (
            "198.51.100.10",
            "http://127.0.0.1:8000",
            None,
        ),
        (
            "127.0.0.1",
            "http://example.com",
            None,
        ),
        (
            "127.0.0.1",
            "http://127.0.0.1:8000",
            "https://example.com",
        ),
    ],
)
def test_nonlocal_or_untrusted_request_rejected(
    monkeypatch,
    peer,
    host,
    origin,
):
    monkeypatch.setenv(
        "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY",
        "1",
    )

    def forbidden_scan(*, limit):
        raise AssertionError(
            "Denied requests must not scan storage."
        )

    monkeypatch.setattr(
        discovery,
        "list_recent_live_runs",
        forbidden_scan,
    )

    headers = {}
    if origin is not None:
        headers["Origin"] = origin

    response = _client(
        peer=peer,
        base_url=host,
    ).get(
        URL,
        headers=headers,
    )

    assert response.status_code == 404


@pytest.mark.parametrize("limit", [0, 51, "invalid"])
def test_invalid_limit_rejected(monkeypatch, limit):
    monkeypatch.setenv(
        "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY",
        "1",
    )

    response = _client().get(
        URL,
        params={"limit": limit},
    )

    assert response.status_code == 422


def test_storage_failure_is_not_exposed(monkeypatch):
    monkeypatch.setenv(
        "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY",
        "1",
    )

    def fail_listing(*, limit):
        raise RuntimeError(
            "/private/internal/run/storage/path"
        )

    monkeypatch.setattr(
        discovery,
        "list_recent_live_runs",
        fail_listing,
    )

    response = _client().get(URL)

    assert response.status_code == 503
    assert "/private/internal/" not in response.text
    assert "no-store" in response.headers[
        "cache-control"
    ]
