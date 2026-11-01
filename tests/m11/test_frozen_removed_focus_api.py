"""Verified, display-only removed-region navigation tests."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.services import (
    frozen_removed_focus_service as focus,
)
from app.backend.services.frozen_slice_service import FrozenSliceError
from app.backend.services.frozen_test_service import FrozenEvidenceError


@pytest.mark.parametrize(
    "case_index,available,count",
    [(1, False, 0), (27, True, 1075), (52, False, 0)],
)
def test_frozen_removed_focus_api(case_index, available, count):
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/frozen-test/cases/{case_index}/removed-focus"
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"

    payload = response.json()

    assert payload["case_index"] == case_index
    assert payload["available"] is available
    assert payload["removed_voxels"] == count
    assert payload["mask_source"] == "removed"
    assert payload["coordinate_system"] == "native_voxel_xyz"

    if available:
        xyz = payload["coordinates"]
        assert isinstance(xyz, dict)
        assert 0 <= xyz["x"] <= 181
        assert 0 <= xyz["y"] <= 217
        assert 0 <= xyz["z"] <= 181
        assert payload["selection"] == (
            "foreground_nearest_removed_centroid"
        )
    else:
        assert payload["coordinates"] is None
        assert payload["selection"] == (
            "unavailable_no_removed_voxels"
        )


@pytest.mark.parametrize("index", [0, 130])
def test_out_of_range(index):
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/frozen-test/cases/{index}/removed-focus"
        )

    assert response.status_code == 422


def synthetic_masks(monkeypatch, baseline, final, removed):
    monkeypatch.setattr(
        focus,
        "load_verified_evaluation",
        lambda: {},
    )
    monkeypatch.setattr(
        focus,
        "_load_mask_arrays",
        lambda index, evaluation: (
            "synthetic",
            {
                "baseline": baseline,
                "final": final,
                "removed": removed,
            },
        ),
    )


def test_empty_removed_mask_is_unavailable(monkeypatch):
    baseline = np.zeros((182, 218, 182), dtype=np.uint8)
    final = np.zeros_like(baseline)
    removed = np.zeros_like(baseline)

    baseline[10, 20, 30] = 1
    final[10, 20, 30] = 1

    synthetic_masks(monkeypatch, baseline, final, removed)

    payload = focus.get_verified_removed_focus(1)

    assert payload["available"] is False
    assert payload["coordinates"] is None
    assert payload["removed_voxels"] == 0


def test_selected_voxel_is_inside_removed_mask(monkeypatch):
    baseline = np.zeros((182, 218, 182), dtype=np.uint8)
    final = np.zeros_like(baseline)
    removed = np.zeros_like(baseline)

    baseline[10, 20, 30] = 1
    baseline[12, 22, 32] = 1

    removed[10, 20, 30] = 1
    removed[12, 22, 32] = 1

    synthetic_masks(monkeypatch, baseline, final, removed)

    payload = focus.get_verified_removed_focus(27)

    assert payload["available"] is True
    assert payload["removed_voxels"] == 2

    xyz = payload["coordinates"]
    assert xyz == {"x": 30, "y": 20, "z": 10}
    assert removed[xyz["z"], xyz["y"], xyz["x"]] == 1


def test_inconsistent_removal_fails(monkeypatch):
    baseline = np.zeros((182, 218, 182), dtype=np.uint8)
    final = np.zeros_like(baseline)
    removed = np.zeros_like(baseline)

    baseline[10, 20, 30] = 1

    synthetic_masks(monkeypatch, baseline, final, removed)

    with pytest.raises(FrozenSliceError):
        focus.get_verified_removed_focus(27)


def test_authentication_failure_returns_503(monkeypatch):
    def unavailable():
        raise FrozenEvidenceError("Synthetic evidence failure.")

    monkeypatch.setattr(
        focus,
        "load_verified_evaluation",
        unavailable,
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/frozen-test/cases/27/removed-focus"
        )

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert "Synthetic evidence failure" not in response.text
