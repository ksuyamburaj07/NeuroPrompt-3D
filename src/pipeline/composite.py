"""Frozen M9C1/M9C3 localized FP compositing primitives.

All arrays in this module use canonical/model XYZ spatial order.

The final frozen strategy is:

    directional_ball15_shell10

support =
    valid SAM FOV
    AND 15 mm physical ball around FP point
    AND 10 mm physical coarse-mask boundary shell

removal =
    coarse foreground
    AND SAM background
    AND support

Only True -> False edits are permitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt


LOCAL_RADIUS_MM = 15.0
BOUNDARY_SHELL_MM = 10.0


class CompositeIntegrityError(ValueError):
    """Hard failure for malformed geometry or compositing inputs."""


def _shape3(
    values: Sequence[int],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.int64,
    )

    if array.shape != (3,):
        raise CompositeIntegrityError(
            f"{name} must contain exactly 3 values."
        )

    if np.any(array <= 0):
        raise CompositeIntegrityError(
            f"{name} values must be positive."
        )

    return array


def _point3(
    values: Sequence[int],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.int64,
    )

    if array.shape != (3,):
        raise CompositeIntegrityError(
            f"{name} must contain exactly 3 values."
        )

    return array


def _spacing3(
    values: Sequence[float],
) -> np.ndarray:
    spacing = np.asarray(
        values,
        dtype=np.float64,
    )

    if spacing.shape != (3,):
        raise CompositeIntegrityError(
            "spacing_xyz_mm must contain exactly 3 values."
        )

    if not np.isfinite(spacing).all():
        raise CompositeIntegrityError(
            "spacing_xyz_mm must be finite."
        )

    if np.any(spacing <= 0.0):
        raise CompositeIntegrityError(
            "spacing_xyz_mm values must be positive."
        )

    return spacing


def array_slices(
    start_xyz: Sequence[int],
    end_xyz_exclusive: Sequence[int],
) -> tuple[slice, slice, slice]:
    """Create three XYZ slices from inclusive start/exclusive end."""

    start = _point3(
        start_xyz,
        name="start_xyz",
    )

    end = _point3(
        end_xyz_exclusive,
        name="end_xyz_exclusive",
    )

    if np.any(end < start):
        raise CompositeIntegrityError(
            "Slice end must not precede slice start."
        )

    return tuple(
        slice(
            int(start[axis]),
            int(end[axis]),
        )
        for axis in range(3)
    )


def _validated_crop_policy(
    full_shape_xyz: Sequence[int],
    crop_policy: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Validate frozen crop/pad source-to-destination metadata."""

    required = (
        "target_shape_xyz",
        "source_start_xyz",
        "source_end_xyz_exclusive",
        "destination_start_xyz",
        "destination_end_xyz_exclusive",
    )

    missing = [
        key
        for key in required
        if key not in crop_policy
    ]

    if missing:
        raise CompositeIntegrityError(
            "Crop policy is missing required fields: "
            + ", ".join(missing)
        )

    full_shape = _shape3(
        full_shape_xyz,
        name="full_shape_xyz",
    )

    target_shape = _shape3(
        crop_policy["target_shape_xyz"],
        name="target_shape_xyz",
    )

    source_start = _point3(
        crop_policy["source_start_xyz"],
        name="source_start_xyz",
    )

    source_end = _point3(
        crop_policy["source_end_xyz_exclusive"],
        name="source_end_xyz_exclusive",
    )

    destination_start = _point3(
        crop_policy["destination_start_xyz"],
        name="destination_start_xyz",
    )

    destination_end = _point3(
        crop_policy["destination_end_xyz_exclusive"],
        name="destination_end_xyz_exclusive",
    )

    if np.any(source_start < 0):
        raise CompositeIntegrityError(
            "source_start_xyz must be non-negative."
        )

    if np.any(destination_start < 0):
        raise CompositeIntegrityError(
            "destination_start_xyz must be non-negative."
        )

    if np.any(source_end > full_shape):
        raise CompositeIntegrityError(
            "Source crop extends outside the full volume."
        )

    if np.any(destination_end > target_shape):
        raise CompositeIntegrityError(
            "Destination crop extends outside the SAM model grid."
        )

    source_extent = (
        source_end
        - source_start
    )

    destination_extent = (
        destination_end
        - destination_start
    )

    if np.any(source_extent <= 0):
        raise CompositeIntegrityError(
            "Crop policy has an empty source extent."
        )

    if not np.array_equal(
        source_extent,
        destination_extent,
    ):
        raise CompositeIntegrityError(
            "Source and destination crop extents must match exactly."
        )

    return {
        "full_shape": full_shape,
        "target_shape": target_shape,
        "source_start": source_start,
        "source_end": source_end,
        "destination_start": destination_start,
        "destination_end": destination_end,
    }


def embed_model_mask_full(
    model_mask_xyz: np.ndarray,
    full_shape_xyz: Sequence[int],
    crop_policy: Mapping[str, object],
) -> np.ndarray:
    """Embed a SAM model-grid mask into canonical full-volume XYZ.

    Padded model-grid voxels are never copied into the full image.
    """

    bounds = _validated_crop_policy(
        full_shape_xyz,
        crop_policy,
    )

    model_mask = np.asarray(
        model_mask_xyz
    )

    if tuple(model_mask.shape) != tuple(
        bounds["target_shape"]
    ):
        raise CompositeIntegrityError(
            "SAM model mask shape does not match target_shape_xyz."
        )

    output = np.zeros(
        tuple(
            int(value)
            for value in bounds[
                "full_shape"
            ]
        ),
        dtype=bool,
    )

    source_slice = array_slices(
        bounds["source_start"],
        bounds["source_end"],
    )

    destination_slice = array_slices(
        bounds["destination_start"],
        bounds["destination_end"],
    )

    output[
        source_slice
    ] = (
        model_mask[
            destination_slice
        ]
        > 0
    )

    return output


def fov_coverage_mask(
    full_shape_xyz: Sequence[int],
    crop_policy: Mapping[str, object],
) -> np.ndarray:
    """Return full-volume mask of source voxels covered by SAM FOV."""

    bounds = _validated_crop_policy(
        full_shape_xyz,
        crop_policy,
    )

    output = np.zeros(
        tuple(
            int(value)
            for value in bounds[
                "full_shape"
            ]
        ),
        dtype=bool,
    )

    output[
        array_slices(
            bounds["source_start"],
            bounds["source_end"],
        )
    ] = True

    return output


def physical_ball_mask(
    shape_xyz: Sequence[int],
    center_xyz: Sequence[int],
    spacing_xyz_mm: Sequence[float],
    radius_mm: float = LOCAL_RADIUS_MM,
) -> np.ndarray:
    """Return an inclusive Euclidean physical-distance ball."""

    shape = _shape3(
        shape_xyz,
        name="shape_xyz",
    )

    center = _point3(
        center_xyz,
        name="center_xyz",
    )

    spacing = _spacing3(
        spacing_xyz_mm
    )

    radius = float(
        radius_mm
    )

    if (
        not np.isfinite(radius)
        or radius < 0.0
    ):
        raise CompositeIntegrityError(
            "radius_mm must be finite and non-negative."
        )

    if (
        np.any(center < 0)
        or np.any(center >= shape)
    ):
        raise CompositeIntegrityError(
            "center_xyz is outside shape_xyz."
        )

    extent = np.ceil(
        radius
        / spacing
    ).astype(
        np.int64
    )

    lower = np.maximum(
        center
        - extent,
        0,
    )

    upper = np.minimum(
        center
        + extent
        + 1,
        shape,
    )

    local_shape = (
        upper
        - lower
    )

    grids = np.ogrid[
        0:int(local_shape[0]),
        0:int(local_shape[1]),
        0:int(local_shape[2]),
    ]

    center_local = (
        center
        - lower
    )

    distance_squared = (
        (
            (
                grids[0]
                - center_local[0]
            )
            * spacing[0]
        )
        ** 2
        +
        (
            (
                grids[1]
                - center_local[1]
            )
            * spacing[1]
        )
        ** 2
        +
        (
            (
                grids[2]
                - center_local[2]
            )
            * spacing[2]
        )
        ** 2
    )

    local_ball = (
        distance_squared
        <= radius ** 2
    )

    output = np.zeros(
        tuple(
            int(value)
            for value in shape
        ),
        dtype=bool,
    )

    output[
        array_slices(
            lower,
            upper,
        )
    ] = local_ball

    return output


def physical_boundary_shell(
    coarse_mask: np.ndarray,
    spacing_xyz_mm: Sequence[float],
    shell_mm: float = BOUNDARY_SHELL_MM,
) -> np.ndarray:
    """Return the frozen two-sided physical coarse-boundary shell."""

    coarse = np.asarray(
        coarse_mask
    ).astype(
        bool
    )

    if coarse.ndim != 3:
        raise CompositeIntegrityError(
            "coarse_mask must be a 3D array."
        )

    spacing = _spacing3(
        spacing_xyz_mm
    )

    shell = float(
        shell_mm
    )

    if (
        not np.isfinite(shell)
        or shell < 0.0
    ):
        raise CompositeIntegrityError(
            "shell_mm must be finite and non-negative."
        )

    inside_distance = (
        distance_transform_edt(
            coarse,
            sampling=tuple(
                float(value)
                for value in spacing
            ),
        )
    )

    outside_distance = (
        distance_transform_edt(
            np.logical_not(
                coarse
            ),
            sampling=tuple(
                float(value)
                for value in spacing
            ),
        )
    )

    return np.logical_or(
        np.logical_and(
            coarse,
            inside_distance
            <= shell,
        ),
        np.logical_and(
            np.logical_not(
                coarse
            ),
            outside_distance
            <= shell,
        ),
    )


def apply_local_fp_composite(
    coarse_mask: np.ndarray,
    fp_sam_mask: np.ndarray,
    valid_sam_fov: np.ndarray,
    fp_point_xyz: Sequence[int],
    spacing_xyz_mm: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the frozen directional_ball15_shell10 FP composite.

    Returns:
        final_mask,
        removal_mask,
        support_mask
    """

    coarse = np.asarray(
        coarse_mask
    ).astype(
        bool
    )

    sam = np.asarray(
        fp_sam_mask
    ).astype(
        bool
    )

    fov = np.asarray(
        valid_sam_fov
    ).astype(
        bool
    )

    if coarse.ndim != 3:
        raise CompositeIntegrityError(
            "coarse_mask must be a 3D array."
        )

    if (
        coarse.shape != sam.shape
        or coarse.shape != fov.shape
    ):
        raise CompositeIntegrityError(
            "coarse_mask, fp_sam_mask, and valid_sam_fov "
            "must have identical 3D shapes."
        )

    ball = physical_ball_mask(
        coarse.shape,
        fp_point_xyz,
        spacing_xyz_mm,
        LOCAL_RADIUS_MM,
    )

    shell = physical_boundary_shell(
        coarse,
        spacing_xyz_mm,
        BOUNDARY_SHELL_MM,
    )

    support = (
        ball
        & shell
        & fov
    )

    removal = (
        coarse
        & np.logical_not(
            sam
        )
        & support
    )

    result = (
        coarse.copy()
    )

    result[
        removal
    ] = False

    return (
        result,
        removal,
        support,
    )
