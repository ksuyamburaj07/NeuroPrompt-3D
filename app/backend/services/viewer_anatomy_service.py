"""Read-only MRI-derived anatomical context surface.

This is a visualization approximation of the nonzero T1N
foreground, not a validated brain-tissue segmentation.
"""

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

from app.backend.services.viewer_mesh_service import (
    MAX_PADDED_VOXELS,
    MAX_SOURCE_VOXELS,
    MAX_TRIANGLES,
    ViewerMeshGeometryError,
    _binary_ply,
)
from app.backend.services.viewer_service import _modality_path


ANATOMY_STEP_VOXELS = 2
ANATOMY_PADDING = 4


def viewer_anatomy_ply(case_id: str, step: int = ANATOMY_STEP_VOXELS) -> bytes:
    """Return a physical-RAS PLY from the staged T1N foreground."""

    if step not in (1, ANATOMY_STEP_VOXELS):
        raise ValueError("Anatomical mesh step must be 1 or 2.")

    path = _modality_path(case_id, "t1n")
    image = nib.load(str(path))

    if (
        len(image.shape) != 3
        or int(np.prod(image.shape)) > MAX_SOURCE_VOXELS
    ):
        raise ViewerMeshGeometryError(
            "Anatomical surface requires a supported 3D MRI."
        )

    affine = np.asarray(image.affine, dtype=np.float64)

    if (
        not np.isfinite(affine).all()
        or abs(np.linalg.det(affine[:3, :3])) < 1e-8
    ):
        raise ViewerMeshGeometryError(
            "Invalid source MRI spatial affine."
        )

    canonical = nib.as_closest_canonical(image)
    values = np.asanyarray(canonical.dataobj)

    if not np.isfinite(values).all():
        raise ViewerMeshGeometryError(
            "Source MRI contains nonfinite values."
        )

    foreground = values != 0

    if not foreground.any():
        raise ViewerMeshGeometryError(
            "No anatomical foreground is available."
        )

    # Crop to observed anatomy without changing its voxel values.
    occupied = np.argwhere(foreground)
    lower = occupied.min(axis=0)
    upper = occupied.max(axis=0) + 1

    crop_shape = upper - lower
    padded_shape = crop_shape + 2 * ANATOMY_PADDING

    if int(np.prod(padded_shape)) > MAX_PADDED_VOXELS:
        raise ViewerMeshGeometryError(
            "Anatomical foreground exceeds the mesh size limit."
        )

    cropped = foreground[
        lower[0]:upper[0],
        lower[1]:upper[1],
        lower[2]:upper[2],
    ]

    padded = np.pad(
        cropped,
        ANATOMY_PADDING,
        mode="constant",
    )

    # Visualization-only sampling; the source MRI is unchanged.
    vertices, faces, _, _ = marching_cubes(
        padded.astype(np.float32),
        level=0.5,
        step_size=step,
        allow_degenerate=False,
    )

    if (
        len(vertices) == 0
        or len(faces) == 0
        or len(faces) > MAX_TRIANGLES
    ):
        raise ViewerMeshGeometryError(
            "Anatomical surface exceeds geometry limits."
        )

    canonical_voxels = (
        vertices + lower - ANATOMY_PADDING
    )

    world_ras = nib.affines.apply_affine(
        canonical.affine,
        canonical_voxels,
    )

    if not np.isfinite(world_ras).all():
        raise ViewerMeshGeometryError(
            "Anatomical surface has invalid world coordinates."
        )

    return _binary_ply(world_ras, faces)
