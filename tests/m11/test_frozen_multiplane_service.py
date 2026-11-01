"""Authenticated frozen coronal/sagittal rendering tests."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.backend.services import (
    frozen_multiplane_service as service,
)
from app.backend.services.frozen_plane_geometry import (
    frozen_mask_plane,
)
from app.backend.services.frozen_test_service import (
    FrozenEvidenceError,
    load_verified_evaluation,
)
from app.backend.services.frozen_slice_service import (
    FrozenSliceError,
    _load_mask_arrays,
)


@pytest.fixture(scope="module")
def frozen_case_027():
    evaluation = load_verified_evaluation()
    _, masks = _load_mask_arrays(27, evaluation)
    return masks


def decoded_png(case_index, plane, index, layer):
    raw = service.render_frozen_nonaxial_png(
        case_index,
        plane,
        index,
        modality="t2f",
        layer=layer,
    )

    with Image.open(BytesIO(raw)) as image:
        image.verify()

    with Image.open(BytesIO(raw)) as image:
        return (
            np.asarray(image.convert("RGB")).copy(),
            image.size,
        )


@pytest.mark.parametrize(
    "plane,index,size,reference_pixel",
    [
        ("coronal", 54, (182, 182), (146, 105)),
        ("sagittal", 105, (218, 182), (146, 54)),
    ],
)
@pytest.mark.parametrize(
    "layer",
    ["baseline", "final", "removed", "support"],
)
def test_authentic_overlay_matches_every_mask_pixel(
    frozen_case_027,
    plane,
    index,
    size,
    reference_pixel,
    layer,
):
    plain, plain_size = decoded_png(
        27, plane, index, "none"
    )
    overlay, overlay_size = decoded_png(
        27, plane, index, layer
    )

    assert plain_size == size
    assert overlay_size == size

    changed = np.any(plain != overlay, axis=2)

    expected = frozen_mask_plane(
        frozen_case_027[layer],
        plane,
        index,
    )

    assert np.array_equal(changed, expected)

    if layer == "removed":
        assert changed[reference_pixel]


@pytest.mark.parametrize(
    "case_index,plane,index,modality,layer",
    [
        (0, "coronal", 54, "t2f", "removed"),
        (130, "coronal", 54, "t2f", "removed"),
        (True, "coronal", 54, "t2f", "removed"),
        (27, "axial", 35, "t2f", "removed"),
        (27, "coronal", -1, "t2f", "removed"),
        (27, "coronal", 218, "t2f", "removed"),
        (27, "sagittal", 182, "t2f", "removed"),
        (27, "sagittal", True, "t2f", "removed"),
        (27, "sagittal", 1.5, "t2f", "removed"),
        (27, "coronal", 54, "invalid", "removed"),
        (27, "coronal", 54, "t2f", "invalid"),
    ],
)
def test_invalid_requests_do_not_open_evidence(
    monkeypatch,
    case_index,
    plane,
    index,
    modality,
    layer,
):
    def forbidden():
        raise AssertionError(
            "Invalid request accessed scientific evidence."
        )

    monkeypatch.setattr(
        service,
        "load_verified_evaluation",
        forbidden,
    )

    with pytest.raises(ValueError):
        service.render_frozen_nonaxial_png(
            case_index,
            plane,
            index,
            modality=modality,
            layer=layer,
        )


def test_historical_unknown_unit_case_renders():
    # Case 052 has four consistent unknown-unit MRI headers.
    _, size = decoded_png(
        52, "coronal", 54, "none"
    )

    assert size == (182, 182)


def test_tampered_mri_manifest_fails_closed(
    tmp_path,
    monkeypatch,
):
    altered = tmp_path / "altered-mri-manifest.txt"
    altered.write_bytes(b"not the frozen MRI manifest")

    monkeypatch.setattr(
        service,
        "MRI_MANIFEST",
        altered,
    )

    with pytest.raises(FrozenSliceError):
        service.render_frozen_nonaxial_png(
            27,
            "coronal",
            54,
            layer="removed",
        )


def test_unavailable_evaluation_fails_closed(monkeypatch):
    def unavailable():
        raise FrozenEvidenceError(
            "Frozen evaluation unavailable."
        )

    monkeypatch.setattr(
        service,
        "load_verified_evaluation",
        unavailable,
    )

    with pytest.raises(FrozenSliceError):
        service.render_frozen_nonaxial_png(
            27,
            "sagittal",
            105,
        )
