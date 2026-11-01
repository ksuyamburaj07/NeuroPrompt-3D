"""Isolated checks for the read-only frozen MRI slice service."""

from hashlib import sha256
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.backend.services import frozen_slice_service as service


@pytest.mark.parametrize(
    "case_index,z,modality,layer",
    [
        (0, 35, "t2f", "removed"),
        (130, 35, "t2f", "removed"),
        (True, 35, "t2f", "removed"),
        (27, -1, "t2f", "removed"),
        (27, 182, "t2f", "removed"),
        (27, 35.5, "t2f", "removed"),
        (27, 35, "invalid", "removed"),
        (27, 35, "t2f", "invalid"),
    ],
)
def test_invalid_request_rejected_before_evidence(
    monkeypatch,
    case_index,
    z,
    modality,
    layer,
):
    def forbidden():
        raise AssertionError(
            "Scientific evidence must not be opened."
        )

    monkeypatch.setattr(
        service,
        "load_verified_evaluation",
        forbidden,
    )

    with pytest.raises(ValueError):
        service.render_frozen_axial_png(
            case_index,
            z,
            modality=modality,
            layer=layer,
        )


def test_verified_source_bytes(tmp_path):
    path = tmp_path / "evidence.bin"
    content = b"immutable-scientific-evidence"

    path.write_bytes(content)
    expected = sha256(content).hexdigest()

    assert service._verified_bytes(path, expected) == content


def test_tampered_source_fails_closed(tmp_path):
    path = tmp_path / "evidence.bin"
    original = b"original"

    path.write_bytes(original)
    expected = sha256(original).hexdigest()

    path.write_bytes(b"modified")

    with pytest.raises(service.FrozenSliceError):
        service._verified_bytes(path, expected)


def test_missing_source_fails_closed(tmp_path):
    missing = tmp_path / "missing.bin"

    with pytest.raises(service.FrozenSliceError):
        service._verified_bytes(missing, "0" * 64)


def test_axial_overlay_uses_native_voxel_coordinates(
    monkeypatch,
):
    """Native XYZ (105,54,35) must map to image XY (105,163)."""

    shape_zyx = (182, 218, 182)

    masks = {
        label: np.zeros(shape_zyx, dtype=bool)
        for label in (
            "baseline", "final", "removed", "support"
        )
    }

    masks["baseline"][35, 54, 105] = True
    masks["removed"][35, 54, 105] = True
    masks["support"][35, 54, 105] = True

    calls = []

    monkeypatch.setattr(
        service,
        "load_verified_evaluation",
        lambda: {"verified": True},
    )

    def fake_masks(index, evaluation):
        assert index == 27
        assert evaluation == {"verified": True}
        return "synthetic-case", masks

    monkeypatch.setattr(
        service,
        "_load_mask_arrays",
        fake_masks,
    )

    def fake_mri(case_id, modality, z):
        calls.append((case_id, modality, z))
        return np.zeros((182, 218), dtype=np.float32)

    monkeypatch.setattr(
        service,
        "_load_verified_mri_slice",
        fake_mri,
    )

    png_without_overlay = service.render_frozen_axial_png(
        27, 35, modality="t2f", layer="none"
    )

    png_with_overlay = service.render_frozen_axial_png(
        27, 35, modality="t2f", layer="removed"
    )

    with Image.open(BytesIO(png_without_overlay)) as img:
        background = np.asarray(img.convert("RGB")).copy()

    with Image.open(BytesIO(png_with_overlay)) as img:
        rendered = np.asarray(img.convert("RGB")).copy()
        assert img.size == (182, 218)

    # Native MRI Y=54 is displayed at row 217-54=163.
    expected_row = 217 - 54
    expected_column = 105

    assert np.array_equal(
        background[expected_row, expected_column],
        [0, 0, 0],
    )

    assert rendered[
        expected_row, expected_column, 0
    ] > 0

    assert np.array_equal(
        rendered[expected_row, 182 - 1 - 105],
        [0, 0, 0],
    )

    changed = np.any(rendered != background, axis=2)

    assert int(changed.sum()) == 1
    assert changed[expected_row, expected_column]

    assert calls == [
        ("synthetic-case", "t2f", 35),
        ("synthetic-case", "t2f", 35),
    ]
