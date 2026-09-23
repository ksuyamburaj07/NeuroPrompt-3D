"""Frozen M9A3/M9B1 SAM adapter geometry.

Conventions
-----------
NeuroPrompt internal coordinates:
    ZYX

NIfTI / canonical spatial arrays:
    XYZ

SAM-Med3D image tensor spatial axes:
    XYZ

Frozen target:
    128 x 128 x 128

The NeuroPrompt adapter never adds the SAM prompt encoder's +0.5
half-voxel offset. That shift remains internal to SAM-Med3D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


SAM_TARGET_SHAPE_XYZ = (
    128,
    128,
    128,
)


class AdapterIntegrityError(ValueError):
    """Hard failure for malformed adapter geometry."""


class PromptPreservingCropInfeasible(ValueError):
    """Semantic valid-case failure: frozen prompts cannot fit in 128^3."""


@dataclass(frozen=True)
class CropPadMetadata:
    """Frozen crop/pad source-to-destination geometry."""

    source_shape_xyz: tuple[int, int, int]
    start_xyz: tuple[int, int, int]
    end_xyz_exclusive: tuple[int, int, int]
    source_start_xyz: tuple[int, int, int]
    source_end_xyz_exclusive: tuple[int, int, int]
    destination_start_xyz: tuple[int, int, int]
    destination_end_xyz_exclusive: tuple[int, int, int]
    target_shape_xyz: tuple[int, int, int]


@dataclass(frozen=True)
class PointTransform:
    """Native NeuroPrompt point mapped into canonical voxel space."""

    native_xyz: tuple[int, int, int]
    world_xyz_mm: tuple[float, float, float]
    canonical_xyz: tuple[int, int, int]
    rounding_residual: float


@dataclass(frozen=True)
class CropWindow:
    """Frozen coarse-centered, prompt-preserving crop window."""

    coarse_center_of_mass_xyz: tuple[float, float, float]
    initial_crop_start_xyz: tuple[int, int, int]
    prompt_preserving_shift_xyz: tuple[int, int, int]
    final_crop_start_xyz: tuple[int, int, int]
    final_crop_end_xyz_exclusive: tuple[int, int, int]
    prompt_min_xyz: tuple[int, int, int]
    prompt_max_xyz: tuple[int, int, int]
    prompt_span_xyz: tuple[int, int, int]


def _vector3_int(
    values: Sequence[int],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.int64,
    )

    if array.shape != (3,):
        raise AdapterIntegrityError(
            f"{name} must contain exactly 3 values."
        )

    return array


def _shape3(
    values: Sequence[int],
    *,
    name: str,
) -> np.ndarray:
    array = _vector3_int(
        values,
        name=name,
    )

    if np.any(
        array <= 0
    ):
        raise AdapterIntegrityError(
            f"{name} values must be positive."
        )

    return array


def _affine44(
    affine,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        affine,
        dtype=np.float64,
    )

    if array.shape != (
        4,
        4,
    ):
        raise AdapterIntegrityError(
            f"{name} must have shape [4,4]."
        )

    if not np.isfinite(
        array
    ).all():
        raise AdapterIntegrityError(
            f"{name} must contain only finite values."
        )

    if (
        abs(
            np.linalg.det(
                array
            )
        )
        < 1e-12
    ):
        raise AdapterIntegrityError(
            f"{name} must be invertible."
        )

    return array


def native_zyx_to_native_xyz(
    point_zyx: Sequence[int],
) -> np.ndarray:
    """Reverse NeuroPrompt ZYX into NIfTI XYZ."""

    point = _vector3_int(
        point_zyx,
        name="point_zyx",
    )

    return np.asarray(
        [
            point[2],
            point[1],
            point[0],
        ],
        dtype=np.int64,
    )


def native_xyz_to_native_zyx(
    point_xyz: Sequence[int],
) -> np.ndarray:
    """Reverse NIfTI XYZ back into NeuroPrompt ZYX."""

    point = _vector3_int(
        point_xyz,
        name="point_xyz",
    )

    return np.asarray(
        [
            point[2],
            point[1],
            point[0],
        ],
        dtype=np.int64,
    )


def apply_affine_point(
    affine,
    point_xyz,
) -> np.ndarray:
    """Apply a 4x4 affine to one XYZ coordinate."""

    matrix = _affine44(
        affine,
        name="affine",
    )

    point = np.asarray(
        point_xyz,
        dtype=np.float64,
    )

    if point.shape != (3,):
        raise AdapterIntegrityError(
            "point_xyz must contain exactly 3 values."
        )

    if not np.isfinite(
        point
    ).all():
        raise AdapterIntegrityError(
            "point_xyz must be finite."
        )

    homogeneous = np.concatenate(
        [
            point,
            np.asarray(
                [1.0],
                dtype=np.float64,
            ),
        ]
    )

    return (
        matrix
        @ homogeneous
    )[:3]


def native_zyx_to_canonical_xyz(
    point_zyx: Sequence[int],
    native_affine,
    canonical_affine,
    *,
    residual_tolerance: float = 1e-4,
) -> PointTransform:
    """Frozen native-ZYX → world → canonical-XYZ transform."""

    native_matrix = _affine44(
        native_affine,
        name="native_affine",
    )

    canonical_matrix = _affine44(
        canonical_affine,
        name="canonical_affine",
    )

    native_xyz = (
        native_zyx_to_native_xyz(
            point_zyx
        )
    )

    world_xyz_mm = (
        apply_affine_point(
            native_matrix,
            native_xyz,
        )
    )

    canonical_xyz_float = (
        apply_affine_point(
            np.linalg.inv(
                canonical_matrix
            ),
            world_xyz_mm,
        )
    )

    canonical_xyz_int = (
        np.rint(
            canonical_xyz_float
        )
        .astype(
            np.int64
        )
    )

    residual = float(
        np.max(
            np.abs(
                canonical_xyz_float
                - canonical_xyz_int
            )
        )
    )

    if (
        not np.isfinite(
            residual
        )
        or residual
        >= float(
            residual_tolerance
        )
    ):
        raise AdapterIntegrityError(
            "Canonical voxel mapping is not integral "
            "within the frozen tolerance."
        )

    return PointTransform(
        native_xyz=tuple(
            int(value)
            for value in native_xyz
        ),
        world_xyz_mm=tuple(
            float(value)
            for value in world_xyz_mm
        ),
        canonical_xyz=tuple(
            int(value)
            for value in canonical_xyz_int
        ),
        rounding_residual=residual,
    )


def compute_prompt_preserving_window(
    coarse_mask_canonical_xyz: np.ndarray,
    prompt_canonical_xyz,
    target_shape_xyz: Sequence[int] = SAM_TARGET_SHAPE_XYZ,
) -> CropWindow:
    """Construct the frozen coarse-COM-centered prompt-preserving window."""

    coarse = np.asarray(
        coarse_mask_canonical_xyz
    ).astype(
        bool
    )

    if coarse.ndim != 3:
        raise AdapterIntegrityError(
            "coarse_mask_canonical_xyz must be 3D."
        )

    if not np.any(
        coarse
    ):
        raise AdapterIntegrityError(
            "Coarse canonical foreground must be non-empty."
        )

    target = _shape3(
        target_shape_xyz,
        name="target_shape_xyz",
    )

    prompts = np.asarray(
        prompt_canonical_xyz,
        dtype=np.int64,
    )

    if (
        prompts.ndim != 2
        or prompts.shape[1] != 3
        or prompts.shape[0] == 0
    ):
        raise AdapterIntegrityError(
            "prompt_canonical_xyz must have shape [N,3] with N >= 1."
        )

    canonical_shape = np.asarray(
        coarse.shape,
        dtype=np.int64,
    )

    if (
        np.any(
            prompts < 0
        )
        or np.any(
            prompts
            >= canonical_shape[
                None,
                :
            ]
        )
    ):
        raise AdapterIntegrityError(
            "Canonical prompt lies outside canonical image geometry."
        )

    coarse_coordinates = np.argwhere(
        coarse
    )

    coarse_center = np.mean(
        coarse_coordinates.astype(
            np.float64
        ),
        axis=0,
    )

    initial_start = np.floor(
        coarse_center
        - (
            target.astype(
                np.float64
            )
            - 1.0
        )
        / 2.0
    ).astype(
        np.int64
    )

    prompt_min = np.min(
        prompts,
        axis=0,
    )

    prompt_max = np.max(
        prompts,
        axis=0,
    )

    prompt_span = (
        prompt_max
        - prompt_min
        + 1
    )

    if np.any(
        prompt_span
        > target
    ):
        raise PromptPreservingCropInfeasible(
            "Frozen AutoPrompt-v2 points do not fit "
            "inside the requested model grid."
        )

    allowed_start_min = (
        prompt_max
        - target
        + 1
    )

    allowed_start_max = (
        prompt_min
    )

    final_start = np.clip(
        initial_start,
        allowed_start_min,
        allowed_start_max,
    ).astype(
        np.int64
    )

    crop_shift = (
        final_start
        - initial_start
    )

    final_end = (
        final_start
        + target
    )

    return CropWindow(
        coarse_center_of_mass_xyz=tuple(
            float(value)
            for value in coarse_center
        ),
        initial_crop_start_xyz=tuple(
            int(value)
            for value in initial_start
        ),
        prompt_preserving_shift_xyz=tuple(
            int(value)
            for value in crop_shift
        ),
        final_crop_start_xyz=tuple(
            int(value)
            for value in final_start
        ),
        final_crop_end_xyz_exclusive=tuple(
            int(value)
            for value in final_end
        ),
        prompt_min_xyz=tuple(
            int(value)
            for value in prompt_min
        ),
        prompt_max_xyz=tuple(
            int(value)
            for value in prompt_max
        ),
        prompt_span_xyz=tuple(
            int(value)
            for value in prompt_span
        ),
    )


def crop_or_pad_xyz(
    array_xyz: np.ndarray,
    start_xyz: Sequence[int],
    target_shape_xyz: Sequence[int] = SAM_TARGET_SHAPE_XYZ,
    *,
    fill_value,
) -> tuple[np.ndarray, CropPadMetadata]:
    """Frozen crop-or-pad operation with explicit source/destination bounds."""

    array = np.asarray(
        array_xyz
    )

    if array.ndim != 3:
        raise AdapterIntegrityError(
            "array_xyz must be 3D."
        )

    source_shape = np.asarray(
        array.shape,
        dtype=np.int64,
    )

    start = _vector3_int(
        start_xyz,
        name="start_xyz",
    )

    target = _shape3(
        target_shape_xyz,
        name="target_shape_xyz",
    )

    end = (
        start
        + target
    )

    source_start = np.maximum(
        start,
        0,
    )

    source_end = np.minimum(
        end,
        source_shape,
    )

    source_extent = (
        source_end
        - source_start
    )

    if np.any(
        source_extent <= 0
    ):
        raise AdapterIntegrityError(
            "Requested crop has no overlap with source volume."
        )

    destination_start = (
        source_start
        - start
    )

    destination_end = (
        destination_start
        + source_extent
    )

    output = np.full(
        tuple(
            int(value)
            for value in target
        ),
        fill_value=fill_value,
        dtype=array.dtype,
    )

    source_slices = tuple(
        slice(
            int(source_start[axis]),
            int(source_end[axis]),
        )
        for axis in range(3)
    )

    destination_slices = tuple(
        slice(
            int(destination_start[axis]),
            int(destination_end[axis]),
        )
        for axis in range(3)
    )

    output[
        destination_slices
    ] = array[
        source_slices
    ]

    metadata = CropPadMetadata(
        source_shape_xyz=tuple(
            int(value)
            for value in source_shape
        ),
        start_xyz=tuple(
            int(value)
            for value in start
        ),
        end_xyz_exclusive=tuple(
            int(value)
            for value in end
        ),
        source_start_xyz=tuple(
            int(value)
            for value in source_start
        ),
        source_end_xyz_exclusive=tuple(
            int(value)
            for value in source_end
        ),
        destination_start_xyz=tuple(
            int(value)
            for value in destination_start
        ),
        destination_end_xyz_exclusive=tuple(
            int(value)
            for value in destination_end
        ),
        target_shape_xyz=tuple(
            int(value)
            for value in target
        ),
    )

    return (
        output,
        metadata,
    )


def canonical_xyz_to_model_xyz(
    canonical_xyz: Sequence[int],
    crop_start_xyz: Sequence[int],
    target_shape_xyz: Sequence[int] = SAM_TARGET_SHAPE_XYZ,
) -> np.ndarray:
    """Frozen canonical-XYZ → SAM model-XYZ point transform."""

    canonical = _vector3_int(
        canonical_xyz,
        name="canonical_xyz",
    )

    start = _vector3_int(
        crop_start_xyz,
        name="crop_start_xyz",
    )

    target = _shape3(
        target_shape_xyz,
        name="target_shape_xyz",
    )

    model_xyz = (
        canonical
        - start
    )

    if (
        np.any(
            model_xyz < 0
        )
        or np.any(
            model_xyz >= target
        )
    ):
        raise AdapterIntegrityError(
            "Canonical point falls outside model grid."
        )

    return model_xyz.astype(
        np.int64
    )


def model_xyz_to_canonical_xyz(
    model_xyz: Sequence[int],
    metadata: CropPadMetadata,
) -> np.ndarray:
    """Map a valid SAM model voxel back into canonical source XYZ.

    Uses the explicit frozen source/destination crop metadata rather
    than assuming destination_start_xyz == (0,0,0).
    """

    model = _vector3_int(
        model_xyz,
        name="model_xyz",
    )

    destination_start = np.asarray(
        metadata.destination_start_xyz,
        dtype=np.int64,
    )

    destination_end = np.asarray(
        metadata.destination_end_xyz_exclusive,
        dtype=np.int64,
    )

    if (
        np.any(
            model < destination_start
        )
        or np.any(
            model >= destination_end
        )
    ):
        raise AdapterIntegrityError(
            "model_xyz lies in padded/out-of-source model space."
        )

    source_start = np.asarray(
        metadata.source_start_xyz,
        dtype=np.int64,
    )

    return (
        source_start
        + (
            model
            - destination_start
        )
    ).astype(
        np.int64
    )


def canonical_xyz_to_native_zyx(
    canonical_xyz: Sequence[int],
    canonical_affine,
    native_affine,
    *,
    residual_tolerance: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Frozen canonical XYZ → world → native ZYX inverse transform."""

    canonical_matrix = _affine44(
        canonical_affine,
        name="canonical_affine",
    )

    native_matrix = _affine44(
        native_affine,
        name="native_affine",
    )

    canonical = _vector3_int(
        canonical_xyz,
        name="canonical_xyz",
    )

    world_xyz_mm = (
        apply_affine_point(
            canonical_matrix,
            canonical,
        )
    )

    native_xyz_float = (
        apply_affine_point(
            np.linalg.inv(
                native_matrix
            ),
            world_xyz_mm,
        )
    )

    native_xyz_int = (
        np.rint(
            native_xyz_float
        )
        .astype(
            np.int64
        )
    )

    residual = float(
        np.max(
            np.abs(
                native_xyz_float
                - native_xyz_int
            )
        )
    )

    if (
        not np.isfinite(
            residual
        )
        or residual
        >= float(
            residual_tolerance
        )
    ):
        raise AdapterIntegrityError(
            "Native voxel round-trip is not integral "
            "within the frozen tolerance."
        )

    return (
        native_xyz_to_native_zyx(
            native_xyz_int
        ),
        residual,
    )


def normalize_positive_model_image(
    image_model_xyz: np.ndarray,
) -> tuple[np.ndarray, float, float, int]:
    """Frozen SAM image normalization over final-crop image > 0."""

    image = np.asarray(
        image_model_xyz,
        dtype=np.float32,
    )

    if image.ndim != 3:
        raise AdapterIntegrityError(
            "image_model_xyz must be 3D."
        )

    if not np.isfinite(
        image
    ).all():
        raise AdapterIntegrityError(
            "image_model_xyz contains non-finite values."
        )

    positive_mask = (
        image
        > 0
    )

    positive_count = int(
        np.count_nonzero(
            positive_mask
        )
    )

    if positive_count == 0:
        raise AdapterIntegrityError(
            "SAM model image has no positive-intensity voxels."
        )

    positive_values = (
        image[
            positive_mask
        ]
        .astype(
            np.float64
        )
    )

    mean = float(
        positive_values.mean()
    )

    std = float(
        positive_values.std(
            ddof=0
        )
    )

    if (
        not np.isfinite(
            mean
        )
        or not np.isfinite(
            std
        )
        or std <= 0.0
    ):
        raise AdapterIntegrityError(
            "Frozen SAM normalization statistics are invalid."
        )

    normalized = np.zeros(
        image.shape,
        dtype=np.float32,
    )

    normalized[
        positive_mask
    ] = (
        (
            image[
                positive_mask
            ].astype(
                np.float64
            )
            - mean
        )
        / std
    ).astype(
        np.float32
    )

    if not np.isfinite(
        normalized
    ).all():
        raise AdapterIntegrityError(
            "Normalized SAM image contains non-finite values."
        )

    return (
        normalized,
        mean,
        std,
        positive_count,
    )
