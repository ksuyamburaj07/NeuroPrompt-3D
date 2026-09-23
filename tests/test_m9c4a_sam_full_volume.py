from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.sam.adapter import (
    crop_or_pad_xyz,
)

from src.sam.full_volume import (
    FullVolumeIntegrityError,
    build_canonical_full_volume_geometry,
    canonical_mask_xyz_to_internal_zyx,
)

from src.sam.preprocessing import (
    prepare_sam_t2f_input,
)


def _synthetic_case(
    tmp_path: Path,
):
    native_shape_xyz = (
        6,
        7,
        8,
    )

    image_xyz = np.zeros(
        native_shape_xyz,
        dtype=np.float32,
    )

    image_xyz[
        1,
        3,
        4,
    ] = 1.0

    image_xyz[
        2,
        3,
        4,
    ] = 3.0

    image_xyz[
        3,
        3,
        4,
    ] = 5.0

    # Native LAS-like geometry:
    # X flips during closest-canonical conversion.
    affine = np.asarray(
        [
            [-1.0, 0.0, 0.0, 5.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    path = (
        tmp_path
        / "synthetic-t2f.nii.gz"
    )

    nib.save(
        nib.Nifti1Image(
            image_xyz,
            affine,
        ),
        str(
            path
        ),
    )

    coarse_zyx = np.zeros(
        (
            8,
            7,
            6,
        ),
        dtype=bool,
    )

    coarse_zyx[
        3:6,
        2:5,
        1:5,
    ] = True

    prepared = prepare_sam_t2f_input(
        path,
        coarse_zyx,
        {
            "uncertainty_hotspot": (
                4,
                3,
                1,
            ),
            "false_positive_correction": (
                4,
                3,
                1,
            ),
            "false_negative_correction": (
                4,
                3,
                0,
            ),
        },
    )

    return (
        coarse_zyx,
        prepared,
    )


def test_full_geometry_has_canonical_shape_and_frozen_spacing(
    tmp_path,
):
    coarse_zyx, prepared = (
        _synthetic_case(
            tmp_path
        )
    )

    geometry = (
        build_canonical_full_volume_geometry(
            coarse_zyx,
            prepared,
        )
    )

    assert (
        geometry
        .coarse_mask_canonical_xyz
        .shape
        == prepared.canonical_shape_xyz
    )

    assert (
        geometry
        .coarse_mask_canonical_xyz
        .dtype
        == bool
    )

    assert np.allclose(
        geometry.spacing_xyz_mm,
        (
            1.0,
            1.5,
            2.0,
        ),
    )


def test_full_canonical_coarse_matches_prepared_model_crop(
    tmp_path,
):
    coarse_zyx, prepared = (
        _synthetic_case(
            tmp_path
        )
    )

    geometry = (
        build_canonical_full_volume_geometry(
            coarse_zyx,
            prepared,
        )
    )

    cropped_uint8, metadata = (
        crop_or_pad_xyz(
            geometry
            .coarse_mask_canonical_xyz
            .astype(
                np.uint8
            ),
            prepared
            .crop_window
            .final_crop_start_xyz,
            fill_value=0,
        )
    )

    assert (
        metadata
        == prepared.crop_metadata
    )

    assert np.array_equal(
        cropped_uint8
        > 0,
        prepared
        .coarse_mask_model_xyz,
    )


def test_canonical_to_internal_roundtrip_is_exact(
    tmp_path,
):
    coarse_zyx, prepared = (
        _synthetic_case(
            tmp_path
        )
    )

    geometry = (
        build_canonical_full_volume_geometry(
            coarse_zyx,
            prepared,
        )
    )

    restored = (
        canonical_mask_xyz_to_internal_zyx(
            geometry
            .coarse_mask_canonical_xyz,
            prepared,
        )
    )

    assert np.array_equal(
        restored,
        coarse_zyx,
    )


def test_single_canonical_landmark_maps_back_to_expected_internal_point(
    tmp_path,
):
    _, prepared = (
        _synthetic_case(
            tmp_path
        )
    )

    canonical_point = (
        prepared
        .canonical_points_xyz[
            "false_positive_correction"
        ]
    )

    mask = np.zeros(
        prepared.canonical_shape_xyz,
        dtype=bool,
    )

    mask[
        canonical_point
    ] = True

    internal = (
        canonical_mask_xyz_to_internal_zyx(
            mask,
            prepared,
        )
    )

    assert (
        int(
            internal.sum()
        )
        == 1
    )

    assert internal[
        4,
        3,
        1,
    ]


def test_shape_mismatch_is_integrity_failure(
    tmp_path,
):
    coarse_zyx, prepared = (
        _synthetic_case(
            tmp_path
        )
    )

    malformed_coarse = (
        coarse_zyx[
            :-1
        ]
    )

    with pytest.raises(
        FullVolumeIntegrityError,
        match="native geometry",
    ):
        build_canonical_full_volume_geometry(
            malformed_coarse,
            prepared,
        )

    malformed_canonical = np.zeros(
        (
            2,
            2,
            2,
        ),
        dtype=bool,
    )

    with pytest.raises(
        FullVolumeIntegrityError,
        match="canonical geometry",
    ):
        canonical_mask_xyz_to_internal_zyx(
            malformed_canonical,
            prepared,
        )
