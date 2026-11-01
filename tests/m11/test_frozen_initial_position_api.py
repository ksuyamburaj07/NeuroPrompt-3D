"""Regression tests for verified initial MRI positioning."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.services import frozen_initial_position_service as positioning
from app.backend.services.frozen_test_service import FrozenEvidenceError


@pytest.mark.parametrize(
    "case_index,expected",
    [
        (1, (128, 117, 68)),
        (27, (49, 85, 55)),
        (52, (113, 156, 85)),
    ],
)
def test_real_frozen_case_position(case_index, expected):
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/frozen-test/cases/{case_index}/initial-position"
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"

    payload = response.json()
    assert payload["case_index"] == case_index
    assert payload["mask_source"] == "final"
    assert payload["coordinate_system"] == "native_voxel_xyz"

    coordinates = payload["coordinates"]
    assert (
        coordinates["x"],
        coordinates["y"],
        coordinates["z"],
    ) == expected


@pytest.mark.parametrize("case_index", [0, 130])
def test_out_of_range_cases_are_rejected(case_index):
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/frozen-test/cases/{case_index}/initial-position"
        )

    assert response.status_code == 422


def test_empty_final_uses_baseline(monkeypatch):
    baseline = np.zeros((182, 218, 182), dtype=np.uint8)
    final = np.zeros_like(baseline)
    baseline[10, 20, 30] = 1

    monkeypatch.setattr(
        positioning,
        "load_verified_evaluation",
        lambda: {},
    )
    monkeypatch.setattr(
        positioning,
        "_load_mask_arrays",
        lambda index, evaluation: (
            "synthetic",
            {"baseline": baseline, "final": final},
        ),
    )

    result = positioning.get_verified_initial_position(1)

    assert result["mask_source"] == "baseline"
    assert result["coordinates"] == {"x": 30, "y": 20, "z": 10}


def test_both_empty_uses_volume_center(monkeypatch):
    empty = np.zeros((182, 218, 182), dtype=np.uint8)

    monkeypatch.setattr(
        positioning,
        "load_verified_evaluation",
        lambda: {},
    )
    monkeypatch.setattr(
        positioning,
        "_load_mask_arrays",
        lambda index, evaluation: (
            "synthetic",
            {"baseline": empty, "final": empty},
        ),
    )

    result = positioning.get_verified_initial_position(1)

    assert result["mask_source"] == "volume_center"
    assert result["coordinates"] == {"x": 91, "y": 109, "z": 91}
    assert result["selection"] == "volume_center_fallback"


def test_failed_evidence_authentication_returns_503(monkeypatch):
    def unavailable():
        raise FrozenEvidenceError("Synthetic verification failure.")

    monkeypatch.setattr(
        positioning,
        "load_verified_evaluation",
        unavailable,
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/frozen-test/cases/27/initial-position"
        )

    assert response.status_code == 503
    assert "Synthetic verification failure" not in response.text
