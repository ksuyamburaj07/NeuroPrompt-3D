"""Display-only anatomical focus metadata regression."""

import numpy as np
import pytest

from app.backend.services import viewer_service


def setup_synthetic(monkeypatch, foreground):
    volume = np.zeros((5, 6, 7), dtype=np.uint8)

    monkeypatch.setattr(
        viewer_service,
        "_case_volume",
        lambda case_id, modality: (
            volume,
            np.eye(4).tolist(),
            ["R", "A", "S"],
        ),
    )

    monkeypatch.setattr(
        viewer_service,
        "_case_source_foreground",
        lambda case_id, modality: foreground,
    )


def test_foreground_bounds_are_exclusive(monkeypatch):
    foreground = np.zeros((5, 6, 7), dtype=bool)
    foreground[1:4, 2:5, 1:6] = True

    setup_synthetic(monkeypatch, foreground)

    data = viewer_service.viewer_metadata("synthetic")

    assert data["shape_ras_xyz"] == [5, 6, 7]
    assert data["foreground_bounds_ras_xyz"] == {
        "lower": [1, 2, 1],
        "upper_exclusive": [4, 5, 6],
    }


def test_empty_foreground_is_rejected(monkeypatch):
    foreground = np.zeros((5, 6, 7), dtype=bool)
    setup_synthetic(monkeypatch, foreground)

    with pytest.raises(ValueError, match="foreground"):
        viewer_service.viewer_metadata("synthetic")


def test_mismatched_foreground_is_rejected(monkeypatch):
    foreground = np.ones((4, 6, 7), dtype=bool)
    setup_synthetic(monkeypatch, foreground)

    with pytest.raises(ValueError, match="foreground"):
        viewer_service.viewer_metadata("synthetic")
