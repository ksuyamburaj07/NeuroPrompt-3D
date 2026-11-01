"""Native-LAS display geometry for authenticated frozen MRI evidence.

MRI arrays use native XYZ order.
Frozen prediction arrays use internal ZYX order.

This module performs only array slicing and exact axis transformations.
It does not load datasets, interpolate, run inference, or alter masks.
"""

import numpy as np


MRI_SHAPE_XYZ = (182, 218, 182)
MASK_SHAPE_ZYX = (182, 218, 182)

PLANE_COUNTS = {
    "axial": 182,
    "coronal": 218,
    "sagittal": 182,
}

# Display directions for the frozen native-LAS affine.
PLANE_DIRECTIONS = {
    "axial": ("R", "L", "A", "P"),
    "coronal": ("R", "L", "S", "I"),
    "sagittal": ("P", "A", "S", "I"),
}


def _validate(
    source,
    expected_shape: tuple[int, int, int],
    plane: str,
    index: int,
) -> None:
    if not isinstance(plane, str) or plane not in PLANE_COUNTS:
        raise ValueError("Unsupported anatomical plane.")

    if type(index) is not int:
        raise ValueError("Slice index must be an integer.")

    if not 0 <= index < PLANE_COUNTS[plane]:
        raise ValueError("Slice index is outside the MRI volume.")

    if tuple(getattr(source, "shape", ())) != expected_shape:
        raise ValueError("Unexpected frozen volume shape.")


def native_mri_plane(
    volume_xyz,
    plane: str,
    index: int,
) -> np.ndarray:
    """Read one native XYZ MRI slice in verified display orientation.

    An NiBabel array proxy may be supplied to avoid loading the volume.
    """
    _validate(volume_xyz, MRI_SHAPE_XYZ, plane, index)

    if plane == "axial":
        native = volume_xyz[:, :, index]
    elif plane == "coronal":
        native = volume_xyz[:, index, :]
    else:
        native = volume_xyz[index, :, :]

    # Orient rows top-to-bottom and columns left-to-right.
    return np.ascontiguousarray(
        np.asarray(native).T[::-1, :]
    )


def frozen_mask_plane(
    mask_zyx: np.ndarray,
    plane: str,
    index: int,
) -> np.ndarray:
    """Read one saved ZYX prediction slice in MRI display orientation."""
    _validate(mask_zyx, MASK_SHAPE_ZYX, plane, index)

    if plane == "axial":
        native = mask_zyx[index, :, :]
    elif plane == "coronal":
        native = mask_zyx[:, index, :]
    else:
        native = mask_zyx[:, :, index]

    return np.ascontiguousarray(
        np.asarray(native)[::-1, :]
    )
