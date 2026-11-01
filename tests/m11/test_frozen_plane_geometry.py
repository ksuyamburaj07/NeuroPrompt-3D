"""Exact native-LAS MRI and frozen-ZYX mask orientation tests."""

import numpy as np
import pytest

from app.backend.services import frozen_plane_geometry as geometry


@pytest.mark.parametrize(
    "plane,index,shape,pixel",
    [
        ("axial", 109, (218, 182), (153, 37)),
        ("coronal", 64, (182, 182), (72, 37)),
        ("sagittal", 37, (182, 218), (72, 64)),
    ],
)
def test_exact_asymmetric_voxel_alignment(
    plane, index, shape, pixel
):
    x, y, z = 37, 64, 109

    mri_xyz = np.zeros(
        geometry.MRI_SHAPE_XYZ, dtype=np.uint8
    )
    mask_zyx = np.zeros(
        geometry.MASK_SHAPE_ZYX, dtype=bool
    )

    mri_xyz[x, y, z] = 255
    mask_zyx[z, y, x] = True

    image = geometry.native_mri_plane(
        mri_xyz, plane, index
    )
    mask = geometry.frozen_mask_plane(
        mask_zyx, plane, index
    )

    assert image.shape == shape
    assert mask.shape == shape

    assert image[pixel] == 255
    assert mask[pixel]

    assert np.count_nonzero(image) == 1
    assert np.count_nonzero(mask) == 1

    assert image.flags.c_contiguous
    assert mask.flags.c_contiguous


class ForbiddenRead:
    """Reject accidental voxel access during input validation."""

    shape = (182, 218, 182)

    def __getitem__(self, item):
        raise AssertionError(
            "Invalid requests must not access voxels."
        )


@pytest.mark.parametrize(
    "plane,index",
    [
        ("oblique", 0),
        ("axial", -1),
        ("axial", 182),
        ("coronal", 218),
        ("sagittal", 182),
        ("axial", True),
        ("axial", 1.5),
    ],
)
def test_invalid_request_rejected_without_voxel_access(
    plane, index
):
    source = ForbiddenRead()

    with pytest.raises(ValueError):
        geometry.native_mri_plane(source, plane, index)

    with pytest.raises(ValueError):
        geometry.frozen_mask_plane(source, plane, index)


def test_invalid_mri_shape():
    incorrect = np.zeros((4, 5, 6), dtype=np.uint8)

    with pytest.raises(ValueError):
        geometry.native_mri_plane(
            incorrect, "axial", 2
        )


def test_invalid_prediction_shape():
    incorrect = np.zeros((4, 5, 6), dtype=np.uint8)

    with pytest.raises(ValueError):
        geometry.frozen_mask_plane(
            incorrect, "sagittal", 2
        )


def test_native_las_direction_labels():
    assert geometry.PLANE_DIRECTIONS == {
        "axial": ("R", "L", "A", "P"),
        "coronal": ("R", "L", "S", "I"),
        "sagittal": ("P", "A", "S", "I"),
    }
