from fastapi.testclient import TestClient

from app.backend.core.constants import (
    FROZEN_RESEARCH_CORE_COMMIT,
    PRE_M11_PROVENANCE_COMMIT,
)
from app.backend.main import app


client = TestClient(app)


def test_health_endpoint_reports_provenance_boundary() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["application"] == "NeuroPrompt-3D"
    assert payload["m9e_read_only"] is True
    assert (
        payload["frozen_research_core_commit"]
        == FROZEN_RESEARCH_CORE_COMMIT
    )
    assert (
        payload["pre_m11_provenance_commit"]
        == PRE_M11_PROVENANCE_COMMIT
    )
