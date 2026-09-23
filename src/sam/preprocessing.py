"""MRI-only frozen SAM-Med3D input preparation.

This module implements the M9A3/M9B1 preprocessing contract without
ground truth.

Input:
    one native BraTS T2-FLAIR NIfTI path,
    deterministic coarse mask in NeuroPrompt internal ZYX,
    frozen crop-influencing automatic prompt points in native ZYX.

Output:
    canonical RAS+ geometry,
    normalized 128^3 T2-FLAIR SAM image in XYZ,
    corresponding deterministic coarse mask in XYZ,
    prompt coordinates in the frozen SAM XYZ model grid,
    and explicit crop/pad provenance.

No segmentation volume is accepted or loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np

from src.sam.adapter import (
    SAM_TARGET_SHAPE_XYZ,
    AdapterIntegrityError,
    CropPadMetadata,
    CropWindow,
    canonical_xyz_to_model_xyz,
    compute_prompt_preserving_window,
    crop_or_pad_xyz,
    model_xyz_to_canonical_xyz,
    native_zyx_to_canonical_xyz,
    normalize_positive_model_image,
)


@dataclass(frozen=True)
class PreparedSAMInput:
    """Complete frozen pre-SAM input package for one case."""

    t2f_path: str

    native_shape_xyz: tuple[int, int, int]
    native_axcodes: tuple[str, str, str]

    canonical_shape_xyz: tuple[int, int, int]
    canonical_axcodes: tuple[str, str, str]

    native_affine: np.ndarray
    canonical_affine: np.ndarray

    image_model_xyz: np.ndarray
    coarse_mask_model_xyz: np.ndarray

    canonical_points_xyz: dict[
        str,
        tuple[int, int, int],
    ]

    model_points_xyz: dict[
        str,
        tuple[int, int, int],
    ]

    crop_window: CropWindow
    crop_metadata: CropPadMetadata

    normalization_mean: float
    normalization_std: float
    normalization_positive_count: int


def _load_native_t2f(
    t2f_path: str | Path,
) -> nib.Nifti1Image:
    """Load exactly one 3D MRI NIfTI supplied by the caller."""

    path = Path(
        t2f_path
    )

    if not path.is_file():
        raise AdapterIntegrityError(
            f"T2-FLAIR NIfTI does not exist: {path}"
        )

    image = nib.load(
        str(
            path
        )
    )

    if len(
        image.shape
    ) != 3:
        raise AdapterIntegrityError(
            "T2-FLAIR NIfTI must be a 3D image."
        )

    affine = np.asarray(
        image.affine,
        dtype=np.float64,
    )

    if (
        affine.shape
        != (
            4,
            4,
        )
        or not np.isfinite(
            affine
        ).all()
    ):
        raise AdapterIntegrityError(
            "T2-FLAIR affine is malformed."
        )

    if (
        abs(
            np.linalg.det(
                affine
            )
        )
        < 1e-12
    ):
        raise AdapterIntegrityError(
            "T2-FLAIR affine must be invertible."
        )

    return image


def _coarse_zyx_to_native_nifti(
    coarse_mask_zyx: np.ndarray,
    native_image: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Place the internal ZYX coarse mask into native NIfTI XYZ geometry."""

    coarse_raw = np.asarray(
        coarse_mask_zyx
    )

    if coarse_raw.ndim != 3:
        raise AdapterIntegrityError(
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
        raise AdapterIntegrityError(
            "coarse_mask_zyx contains non-finite values."
        )

    native_shape_xyz = tuple(
        int(value)
        for value in native_image.shape
    )

    expected_zyx_shape = (
        native_shape_xyz[
            2
        ],
        native_shape_xyz[
            1
        ],
        native_shape_xyz[
            0
        ],
    )

    if (
        tuple(
            coarse_raw.shape
        )
        != expected_zyx_shape
    ):
        raise AdapterIntegrityError(
            "coarse_mask_zyx geometry does not match "
            "the native T2-FLAIR NIfTI."
        )

    coarse_native_xyz = np.transpose(
        coarse_raw.astype(
            np.uint8
        ),
        (
            2,
            1,
            0,
        ),
    )

    return nib.Nifti1Image(
        coarse_native_xyz,
        np.asarray(
            native_image.affine,
            dtype=np.float64,
        ),
    )


def prepare_sam_t2f_input(
    t2f_path: str | Path,
    coarse_mask_zyx: np.ndarray,
    crop_prompt_points_zyx: Mapping[
        str,
        Sequence[int],
    ],
) -> PreparedSAMInput:
    """Prepare the frozen 128^3 single-channel SAM input.

    `crop_prompt_points_zyx` must contain only the frozen points that
    are intended to influence the prompt-preserving crop.

    For the frozen AutoPrompt-v2 geometry this means:
        uncertainty hotspot,
        false-positive correction point,
        false-negative correction point.

    The historical control-positive point must not be supplied by the
    final automatic pipeline.
    """

    if not crop_prompt_points_zyx:
        raise AdapterIntegrityError(
            "At least one crop-influencing prompt point is required."
        )

    native_image = _load_native_t2f(
        t2f_path
    )

    coarse_native = (
        _coarse_zyx_to_native_nifti(
            coarse_mask_zyx,
            native_image,
        )
    )

    image_canonical = (
        nib.as_closest_canonical(
            native_image
        )
    )

    coarse_canonical = (
        nib.as_closest_canonical(
            coarse_native
        )
    )

    if (
        image_canonical.shape
        != coarse_canonical.shape
    ):
        raise AdapterIntegrityError(
            "Canonical image/coarse-mask shapes differ."
        )

    if not np.allclose(
        image_canonical.affine,
        coarse_canonical.affine,
        atol=1e-5,
        rtol=0.0,
    ):
        raise AdapterIntegrityError(
            "Canonical image/coarse-mask affines differ."
        )

    native_axcodes = tuple(
        str(value)
        for value in nib.aff2axcodes(
            native_image.affine
        )
    )

    canonical_axcodes = tuple(
        str(value)
        for value in nib.aff2axcodes(
            image_canonical.affine
        )
    )

    if (
        canonical_axcodes
        != (
            "R",
            "A",
            "S",
        )
    ):
        raise AdapterIntegrityError(
            "Canonicalized T2-FLAIR must be RAS+."
        )

    image_canonical_xyz = np.asarray(
        image_canonical.dataobj,
        dtype=np.float32,
    )

    coarse_canonical_xyz = (
        np.asarray(
            coarse_canonical.dataobj
        )
        > 0
    )

    if not np.isfinite(
        image_canonical_xyz
    ).all():
        raise AdapterIntegrityError(
            "T2-FLAIR contains non-finite values."
        )

    if not np.any(
        coarse_canonical_xyz
    ):
        raise AdapterIntegrityError(
            "Canonical deterministic coarse foreground is empty."
        )

    canonical_points_xyz = {}

    for point_name, point_zyx in (
        crop_prompt_points_zyx.items()
    ):
        mapped = (
            native_zyx_to_canonical_xyz(
                point_zyx,
                native_image.affine,
                image_canonical.affine,
            )
        )

        canonical = tuple(
            int(value)
            for value in mapped.canonical_xyz
        )

        canonical_shape = np.asarray(
            image_canonical_xyz.shape,
            dtype=np.int64,
        )

        canonical_array = np.asarray(
            canonical,
            dtype=np.int64,
        )

        if (
            np.any(
                canonical_array < 0
            )
            or np.any(
                canonical_array
                >= canonical_shape
            )
        ):
            raise AdapterIntegrityError(
                f"Canonical prompt lies outside image: "
                f"{point_name!r}"
            )

        canonical_points_xyz[
            str(
                point_name
            )
        ] = canonical

    crop_window = (
        compute_prompt_preserving_window(
            coarse_canonical_xyz,
            np.asarray(
                list(
                    canonical_points_xyz.values()
                ),
                dtype=np.int64,
            ),
            target_shape_xyz=(
                SAM_TARGET_SHAPE_XYZ
            ),
        )
    )

    crop_start_xyz = (
        crop_window
        .final_crop_start_xyz
    )

    image_model_raw_xyz, image_metadata = (
        crop_or_pad_xyz(
            image_canonical_xyz,
            crop_start_xyz,
            SAM_TARGET_SHAPE_XYZ,
            fill_value=0.0,
        )
    )

    coarse_model_uint8, coarse_metadata = (
        crop_or_pad_xyz(
            coarse_canonical_xyz.astype(
                np.uint8
            ),
            crop_start_xyz,
            SAM_TARGET_SHAPE_XYZ,
            fill_value=0,
        )
    )

    if (
        image_metadata
        != coarse_metadata
    ):
        raise AdapterIntegrityError(
            "Image/coarse crop metadata differ unexpectedly."
        )

    coarse_model_xyz = (
        coarse_model_uint8
        > 0
    )

    (
        image_model_xyz,
        normalization_mean,
        normalization_std,
        normalization_positive_count,
    ) = normalize_positive_model_image(
        image_model_raw_xyz
    )

    model_points_xyz = {}

    for point_name, canonical_xyz in (
        canonical_points_xyz.items()
    ):
        model_xyz = (
            canonical_xyz_to_model_xyz(
                canonical_xyz,
                crop_start_xyz,
                SAM_TARGET_SHAPE_XYZ,
            )
        )

        # Explicitly verify that the point is in real source coverage,
        # not merely inside padded 128^3 model space.
        roundtrip_canonical = (
            model_xyz_to_canonical_xyz(
                model_xyz,
                image_metadata,
            )
        )

        if not np.array_equal(
            roundtrip_canonical,
            np.asarray(
                canonical_xyz,
                dtype=np.int64,
            ),
        ):
            raise AdapterIntegrityError(
                "Model/canonical prompt round-trip failed."
            )

        model_points_xyz[
            point_name
        ] = tuple(
            int(value)
            for value in model_xyz
        )

    return PreparedSAMInput(
        t2f_path=str(
            Path(
                t2f_path
            )
        ),
        native_shape_xyz=tuple(
            int(value)
            for value in native_image.shape
        ),
        native_axcodes=native_axcodes,
        canonical_shape_xyz=tuple(
            int(value)
            for value in image_canonical_xyz.shape
        ),
        canonical_axcodes=canonical_axcodes,
        native_affine=np.asarray(
            native_image.affine,
            dtype=np.float64,
        ).copy(),
        canonical_affine=np.asarray(
            image_canonical.affine,
            dtype=np.float64,
        ).copy(),
        image_model_xyz=(
            image_model_xyz
        ),
        coarse_mask_model_xyz=(
            coarse_model_xyz
        ),
        canonical_points_xyz=(
            canonical_points_xyz
        ),
        model_points_xyz=(
            model_points_xyz
        ),
        crop_window=(
            crop_window
        ),
        crop_metadata=(
            image_metadata
        ),
        normalization_mean=float(
            normalization_mean
        ),
        normalization_std=float(
            normalization_std
        ),
        normalization_positive_count=int(
            normalization_positive_count
        ),
    )
