"""Legacy compatibility and shared-volume MRI contrast regression."""

from hashlib import sha256
from io import BytesIO

import pytest
from PIL import Image

from app.backend.services import (
    frozen_slice_service as axial,
    frozen_multiplane_service as nonaxial,
)


CASES = [
    (
        "axial", 35, (163, 105),
        "06d50726df01f41bb7e3daa0d1fa034"
        "ef753fd45eab3dff2a3a639c584046bde",
    ),
    (
        "coronal", 54, (146, 105),
        "e4164d5fede22dc5fae41392bb28f37"
        "af03c62feb4edd346cd341f46f9c369c5",
    ),
    (
        "sagittal", 105, (146, 54),
        "ccefee18364d0676d41ab0bc0729adc"
        "bee271128e99f93eb52cfc4107d2578d0",
    ),
]


def render(plane, index, contrast=None):
    options = {"modality": "t2f", "layer": "none"}

    if contrast is not None:
        options["contrast"] = contrast

    if plane == "axial":
        return axial.render_frozen_axial_png(
            27, index, **options
        )

    return nonaxial.render_frozen_nonaxial_png(
        27, plane, index, **options
    )


@pytest.mark.parametrize("plane,index,pixel,digest", CASES)
def test_default_and_explicit_legacy_match_frozen_fingerprint(
    plane, index, pixel, digest
):
    default = render(plane, index)
    explicit = render(plane, index, "slice")

    assert default == explicit
    assert sha256(default).hexdigest() == digest


@pytest.mark.parametrize("plane,index,pixel,digest", CASES)
def test_shared_contrast_reference_voxel(
    plane, index, pixel, digest
):
    data = render(plane, index, "volume")

    row, column = pixel

    with Image.open(BytesIO(data)) as image:
        assert image.convert("RGB").getpixel(
            (column, row)
        ) == (207, 207, 207)


@pytest.mark.parametrize("plane,index,pixel,digest", CASES)
def test_invalid_contrast_rejected_before_evidence_access(
    plane, index, pixel, digest, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Invalid contrast accessed frozen evidence."
        )

    monkeypatch.setattr(
        axial, "load_verified_evaluation", forbidden
    )
    monkeypatch.setattr(
        nonaxial, "load_verified_evaluation", forbidden
    )

    with pytest.raises(ValueError):
        render(plane, index, "invalid")
