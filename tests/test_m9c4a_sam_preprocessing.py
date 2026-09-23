from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.sam.adapter import (
    AdapterIntegrityError,
    model_xyz_to_canonical_xyz,
)

from src.sam.preprocessing import (
    prepare_sam_t2f_input,
)


def _write_synthetic_t2f(
    tmp_path: Path,
):
    """Create one small non-RAS synthetic MRI only."""

    native_shape_xyz = (
        6,
        7,
        8,
    )

    data_xyz = np.zeros(
        native_shape_xyz,
        dtype=np.float32,
    )

    # Three positive values so frozen normalization statistics are
    # exactly testable.
    data_xyz[
        1,
        3,
        4,
    ] = 1.0

    data_xyz[
        2,
        3,
        4,
    ] = 3.0

    data_xyz[
        3,
        3,
        4,
    ] = 5.0

    # Native X points toward L, so closest-canonical must flip X.
    affine = np.asarray(
        [
            [-1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, 0.0],
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
            data_xyz,
            affine,
        ),
        str(
            path
        ),
    )

    return (
        path,
        data_xyz,
        affine,
    )


def _synthetic_coarse_zyx():
    # Native XYZ shape is (6,7,8), therefore internal is ZYX (8,7,6).
    coarse = np.zeros(
        (
            8,
            7,
            6,
        ),
        dtype=bool,
    )

    coarse[
        3:6,
        2:5,
        1:5,
    ] = True

    return coarse


def test_synthetic_preparation_is_mri_only_and_ras(tmp_path):
    path, _, _ = (
        _write_synthetic_t2f(
            tmp_path
        )
    )

    coarse = (
        _synthetic_coarse_zyx()
    )

    points = {
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
    }

    prepared = prepare_sam_t2f_input(
        path,
        coarse,
        points,
    )

    assert (
        prepared.native_axcodes
        == (
            "L",
            "A",
            "S",
        )
    )

    assert (
        prepared.canonical_axcodes
        == (
            "R",
            "A",
            "S",
        )
    )

    assert prepared.image_model_xyz.shape == (
        128,
        128,
        128,
    )

    assert prepared.coarse_mask_model_xyz.shape == (
        128,
        128,
        128,
    )

    assert prepared.image_model_xyz.dtype == np.float32
    assert prepared.coarse_mask_model_xyz.dtype == bool


def test_native_zyx_prompt_maps_through_real_canonicalization(tmp_path):
    path, _, _ = (
        _write_synthetic_t2f(
            tmp_path
        )
    )

    coarse = (
        _synthetic_coarse_zyx()
    )

    native_point_zyx = (
        4,
        3,
        1,
    )

    prepared = prepare_sam_t2f_input(
        path,
        coarse,
        {
            "fp": (
                native_point_zyx
            ),
        },
    )

    # Native ZYX (4,3,1) -> native XYZ (1,3,4).
    # X is flipped by closest-canonical in a 6-voxel X axis:
    # canonical X = 5 - native X = 4.
    assert (
        prepared.canonical_points_xyz[
            "fp"
        ]
        == (
            4,
            3,
            4,
        )
    )

    model_xyz = (
        prepared.model_points_xyz[
            "fp"
        ]
    )

    canonical_roundtrip = (
        model_xyz_to_canonical_xyz(
            model_xyz,
            prepared.crop_metadata,
        )
    )

    assert np.array_equal(
        canonical_roundtrip,
        np.asarray(
            (
                4,
                3,
                4,
            )
        ),
    )


def test_coarse_mask_uses_same_orientation_transform_as_t2f(tmp_path):
    path, _, _ = (
        _write_synthetic_t2f(
            tmp_path
        )
    )

    coarse = (
        _synthetic_coarse_zyx()
    )

    point_zyx = (
        4,
        3,
        1,
    )

    assert coarse[
        point_zyx
    ]

    prepared = prepare_sam_t2f_input(
        path,
        coarse,
        {
            "fp": (
                point_zyx
            ),
        },
    )

    model_point = (
        prepared.model_points_xyz[
            "fp"
        ]
    )

    assert prepared.coarse_mask_model_xyz[
        model_point
    ]


def test_small_native_image_exercises_real_leading_padding(tmp_path):
    path, _, _ = (
        _write_synthetic_t2f(
            tmp_path
        )
    )

    coarse = (
        _synthetic_coarse_zyx()
    )

    prepared = prepare_sam_t2f_input(
        path,
        coarse,
        {
            "hotspot": (
                4,
                3,
                1,
            ),
            "fp": (
                4,
                3,
                1,
            ),
            "fn": (
                4,
                3,
                0,
            ),
        },
    )

    destination_start = np.asarray(
        prepared
        .crop_metadata
        .destination_start_xyz,
        dtype=np.int64,
    )

    # The synthetic source is much smaller than 128^3, so model-space
    # leading padding must genuinely be present.
    assert np.any(
        destination_start
        > 0
    )

    for name, model_point in (
        prepared.model_points_xyz.items()
    ):
        recovered = (
            model_xyz_to_canonical_xyz(
                model_point,
                prepared.crop_metadata,
            )
        )

        assert np.array_equal(
            recovered,
            np.asarray(
                prepared
                .canonical_points_xyz[
                    name
                ]
            ),
        )


def test_normalization_matches_frozen_positive_voxel_rule(tmp_path):
    path, _, _ = (
        _write_synthetic_t2f(
            tmp_path
        )
    )

    coarse = (
        _synthetic_coarse_zyx()
    )

    prepared = prepare_sam_t2f_input(
        path,
        coarse,
        {
            "fp": (
                4,
                3,
                1,
            ),
        },
    )

    expected = np.asarray(
        [
            1.0,
            3.0,
            5.0,
        ],
        dtype=np.float64,
    )

    assert (
        prepared
        .normalization_positive_count
        == 3
    )

    assert np.isclose(
        prepared.normalization_mean,
        expected.mean(),
    )

    assert np.isclose(
        prepared.normalization_std,
        expected.std(
            ddof=0
        ),
    )

    # All model padding begins as zero and remains zero because the
    # normalization writes only into original image > 0 locations.
    destination_start = np.asarray(
        prepared
        .crop_metadata
        .destination_start_xyz,
        dtype=np.int64,
    )

    if destination_start[0] > 0:
        assert np.all(
            prepared.image_model_xyz[
                :destination_start[0],
                :,
                :,
            ]
            == 0.0
        )


def test_coarse_shape_mismatch_is_integrity_failure(tmp_path):
    path, _, _ = (
        _write_synthetic_t2f(
            tmp_path
        )
    )

    malformed = np.zeros(
        (
            8,
            7,
            5,
        ),
        dtype=bool,
    )

    with pytest.raises(
        AdapterIntegrityError,
        match="does not match",
    ):
        prepare_sam_t2f_input(
            path,
            malformed,
            {
                "fp": (
                    4,
                    3,
                    1,
                ),
            },
        )


def test_nonfinite_t2f_is_integrity_failure(tmp_path):
    shape = (
        6,
        7,
        8,
    )

    image = np.ones(
        shape,
        dtype=np.float32,
    )

    image[
        1,
        1,
        1,
    ] = np.nan

    path = (
        tmp_path
        / "synthetic-nonfinite-t2f.nii.gz"
    )

    nib.save(
        nib.Nifti1Image(
            image,
            np.eye(
                4,
                dtype=np.float64,
            ),
        ),
        str(
            path
        ),
    )

    coarse = np.ones(
        (
            8,
            7,
            6,
        ),
        dtype=bool,
    )

    with pytest.raises(
        AdapterIntegrityError,
        match="non-finite",
    ):
        prepare_sam_t2f_input(
            path,
            coarse,
            {
                "fp": (
                    1,
                    1,
                    1,
                ),
            },
        )


def test_no_prompt_points_is_integrity_failure(tmp_path):
    path, _, _ = (
        _write_synthetic_t2f(
            tmp_path
        )
    )

    coarse = (
        _synthetic_coarse_zyx()
    )

    with pytest.raises(
        AdapterIntegrityError,
        match="At least one",
    ):
        prepare_sam_t2f_input(
            path,
            coarse,
            {},
        )


def test_missing_t2f_file_is_integrity_failure(tmp_path):
    missing = (
        tmp_path
        / "does-not-exist-t2f.nii.gz"
    )

    coarse = (
        _synthetic_coarse_zyx()
    )

    with pytest.raises(
        AdapterIntegrityError,
        match="does not exist",
    ):
        prepare_sam_t2f_input(
            missing,
            coarse,
            {
                "fp": (
                    4,
                    3,
                    1,
                ),
            },
        )
