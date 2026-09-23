"""Full-volume orientation bridge for the frozen SAM adapter.

The deterministic NeuroPrompt baseline uses internal spatial order ZYX.
The SAM adapter/compositor operates in closest-canonical NIfTI XYZ.

This module performs only exact orientation permutation/flip operations.
There is no interpolation, resampling, MRI loading, or inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np

from src.sam.preprocessing import PreparedSAMInput


class FullVolumeIntegrityError(ValueError):
    """Hard failure for malformed full-volume geometry."""


@dataclass(frozen=True)
class CanonicalFullVolumeGeometry:
    """Full canonical geometry needed by frozen SAM compositing."""

    coarse_mask_canonical_xyz: np.ndarray
    spacing_xyz_mm: tuple[float, float, float]


def _affine44(
    affine,
    *,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(
        affine,
        dtype=np.float64,
    )

    if matrix.shape != (
        4,
        4,
    ):
        raise FullVolumeIntegrityError(
            f"{name} must have shape [4,4]."
        )

    if not np.isfinite(
        matrix
    ).all():
        raise FullVolumeIntegrityError(
            f"{name} must contain finite values."
        )

    if abs(
        np.linalg.det(
            matrix
        )
    ) < 1e-12:
        raise FullVolumeIntegrityError(
            f"{name} must be invertible."
        )

    return matrix


def _orientation_transform(
    source_affine,
    destination_affine,
) -> np.ndarray:
    """Return permutation/flip transform from source axes to destination."""

    source = _affine44(
        source_affine,
        name="source_affine",
    )

    destination = _affine44(
        destination_affine,
        name="destination_affine",
    )

    source_orientation = (
        nib.orientations.io_orientation(
            source
        )
    )

    destination_orientation = (
        nib.orientations.io_orientation(
            destination
        )
    )

    if (
        source_orientation.shape
        != (
            3,
            2,
        )
        or destination_orientation.shape
        != (
            3,
            2,
        )
        or not np.isfinite(
            source_orientation
        ).all()
        or not np.isfinite(
            destination_orientation
        ).all()
    ):
        raise FullVolumeIntegrityError(
            "Unexpected NIfTI orientation geometry."
        )

    return nib.orientations.ornt_transform(
        source_orientation,
        destination_orientation,
    )


def canonical_spacing_xyz_mm(
    canonical_affine,
) -> tuple[float, float, float]:
    """Return frozen physical voxel spacing from canonical affine.

    This is the exact column-norm definition used by the frozen
    M9B3/M9C1 full-volume characterization code.
    """

    affine = _affine44(
        canonical_affine,
        name="canonical_affine",
    )

    spacing = np.sqrt(
        np.sum(
            affine[
                :3,
                :3,
            ]
            ** 2,
            axis=0,
        )
    )

    if (
        spacing.shape
        != (
            3,
        )
        or not np.isfinite(
            spacing
        ).all()
        or np.any(
            spacing <= 0.0
        )
    ):
        raise FullVolumeIntegrityError(
            "Canonical voxel spacing is invalid."
        )

    return tuple(
        float(value)
        for value in spacing
    )


def build_canonical_full_volume_geometry(
    coarse_mask_zyx: np.ndarray,
    prepared: PreparedSAMInput,
) -> CanonicalFullVolumeGeometry:
    """Map deterministic coarse mask from internal ZYX to canonical XYZ."""

    coarse_raw = np.asarray(
        coarse_mask_zyx
    )

    if coarse_raw.ndim != 3:
        raise FullVolumeIntegrityError(
            "coarse_mask_zyx must be 3D."
        )

    if (
        np.issubdtype(
            coarse_raw.dtype,
            np.number,
        )
        and not np.isfinite(
            coarse_raw
        ).all()
    ):
        raise FullVolumeIntegrityError(
            "coarse_mask_zyx contains non-finite values."
        )

    expected_internal_shape = (
        prepared.native_shape_xyz[
            2
        ],
        prepared.native_shape_xyz[
            1
        ],
        prepared.native_shape_xyz[
            0
        ],
    )

    if tuple(
        coarse_raw.shape
    ) != expected_internal_shape:
        raise FullVolumeIntegrityError(
            "coarse_mask_zyx does not match prepared native geometry."
        )

    coarse_native_xyz = np.transpose(
        coarse_raw.astype(
            bool
        ),
        (
            2,
            1,
            0,
        ),
    )

    native_to_canonical = (
        _orientation_transform(
            prepared.native_affine,
            prepared.canonical_affine,
        )
    )

    coarse_canonical_xyz = (
        nib.orientations.apply_orientation(
            coarse_native_xyz,
            native_to_canonical,
        )
    )

    if tuple(
        coarse_canonical_xyz.shape
    ) != tuple(
        prepared.canonical_shape_xyz
    ):
        raise FullVolumeIntegrityError(
            "Canonical coarse-mask geometry does not match prepared image."
        )

    coarse_canonical_xyz = (
        np.ascontiguousarray(
            coarse_canonical_xyz,
            dtype=bool,
        )
    )

    spacing_xyz_mm = (
        canonical_spacing_xyz_mm(
            prepared.canonical_affine
        )
    )

    return CanonicalFullVolumeGeometry(
        coarse_mask_canonical_xyz=(
            coarse_canonical_xyz
        ),
        spacing_xyz_mm=(
            spacing_xyz_mm
        ),
    )


def canonical_mask_xyz_to_internal_zyx(
    mask_canonical_xyz: np.ndarray,
    prepared: PreparedSAMInput,
) -> np.ndarray:
    """Restore full canonical XYZ mask to NeuroPrompt internal ZYX."""

    mask_raw = np.asarray(
        mask_canonical_xyz
    )

    if mask_raw.ndim != 3:
        raise FullVolumeIntegrityError(
            "mask_canonical_xyz must be 3D."
        )

    if (
        np.issubdtype(
            mask_raw.dtype,
            np.number,
        )
        and not np.isfinite(
            mask_raw
        ).all()
    ):
        raise FullVolumeIntegrityError(
            "mask_canonical_xyz contains non-finite values."
        )

    if tuple(
        mask_raw.shape
    ) != tuple(
        prepared.canonical_shape_xyz
    ):
        raise FullVolumeIntegrityError(
            "Canonical mask does not match prepared canonical geometry."
        )

    canonical_to_native = (
        _orientation_transform(
            prepared.canonical_affine,
            prepared.native_affine,
        )
    )

    native_xyz = (
        nib.orientations.apply_orientation(
            mask_raw.astype(
                bool
            ),
            canonical_to_native,
        )
    )

    if tuple(
        native_xyz.shape
    ) != tuple(
        prepared.native_shape_xyz
    ):
        raise FullVolumeIntegrityError(
            "Restored native mask geometry is unexpected."
        )

    internal_zyx = np.transpose(
        native_xyz,
        (
            2,
            1,
            0,
        ),
    )

    return np.ascontiguousarray(
        internal_zyx,
        dtype=bool,
    )
