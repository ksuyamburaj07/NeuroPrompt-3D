import numpy as np
import pytest

from src.sam.adapter import (
    SAM_TARGET_SHAPE_XYZ,
    AdapterIntegrityError,
    PromptPreservingCropInfeasible,
    canonical_xyz_to_model_xyz,
    canonical_xyz_to_native_zyx,
    compute_prompt_preserving_window,
    crop_or_pad_xyz,
    model_xyz_to_canonical_xyz,
    native_zyx_to_canonical_xyz,
    native_zyx_to_native_xyz,
    normalize_positive_model_image,
)


def test_frozen_target_shape():
    assert SAM_TARGET_SHAPE_XYZ == (
        128,
        128,
        128,
    )


def test_zyx_to_xyz_reverses_axes():
    result = native_zyx_to_native_xyz(
        (
            7,
            11,
            13,
        )
    )

    assert np.array_equal(
        result,
        np.asarray(
            (
                13,
                11,
                7,
            )
        ),
    )


def test_native_to_canonical_affine_mapping_and_roundtrip():
    # Synthetic pure reorientation-like transform.
    native_affine = np.asarray(
        [
            [-1.0, 0.0, 0.0, 20.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    canonical_affine = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    original_zyx = (
        4,
        6,
        8,
    )

    mapped = native_zyx_to_canonical_xyz(
        original_zyx,
        native_affine,
        canonical_affine,
    )

    assert mapped.native_xyz == (
        8,
        6,
        4,
    )

    assert mapped.canonical_xyz == (
        12,
        6,
        4,
    )

    roundtrip, residual = (
        canonical_xyz_to_native_zyx(
            mapped.canonical_xyz,
            canonical_affine,
            native_affine,
        )
    )

    assert np.array_equal(
        roundtrip,
        np.asarray(
            original_zyx
        ),
    )

    assert residual < 1e-4


def test_coarse_centered_initial_crop_formula():
    coarse = np.zeros(
        (
            200,
            200,
            200,
        ),
        dtype=bool,
    )

    # Single voxel gives exact COM.
    coarse[
        100,
        100,
        100,
    ] = True

    prompts = np.asarray(
        [
            (
                100,
                100,
                100,
            ),
        ],
        dtype=np.int64,
    )

    window = compute_prompt_preserving_window(
        coarse,
        prompts,
    )

    expected = tuple(
        np.floor(
            np.asarray(
                (
                    100.0,
                    100.0,
                    100.0,
                )
            )
            - (
                np.asarray(
                    SAM_TARGET_SHAPE_XYZ,
                    dtype=np.float64,
                )
                - 1.0
            )
            / 2.0
        ).astype(
            np.int64
        )
    )

    assert (
        window.initial_crop_start_xyz
        == expected
    )


def test_prompt_preserving_crop_translates_minimally():
    coarse = np.zeros(
        (
            220,
            220,
            220,
        ),
        dtype=bool,
    )

    coarse[
        100,
        100,
        100,
    ] = True

    # Initial coarse-centered start is 36.
    # Point X=170 requires start >= 43 to fit in 128 voxels.
    prompts = np.asarray(
        [
            (
                100,
                100,
                100,
            ),
            (
                170,
                100,
                100,
            ),
        ],
        dtype=np.int64,
    )

    window = compute_prompt_preserving_window(
        coarse,
        prompts,
    )

    assert (
        window.initial_crop_start_xyz[0]
        == 36
    )

    assert (
        window.final_crop_start_xyz[0]
        == 43
    )

    assert (
        window.prompt_preserving_shift_xyz[0]
        == 7
    )


def test_infeasible_prompt_span_is_semantic_failure():
    coarse = np.zeros(
        (
            250,
            250,
            250,
        ),
        dtype=bool,
    )

    coarse[
        100,
        100,
        100,
    ] = True

    prompts = np.asarray(
        [
            (
                0,
                100,
                100,
            ),
            (
                128,
                100,
                100,
            ),
        ],
        dtype=np.int64,
    )

    with pytest.raises(
        PromptPreservingCropInfeasible,
        match="do not fit",
    ):
        compute_prompt_preserving_window(
            coarse,
            prompts,
        )


def test_crop_or_pad_leading_padding_metadata():
    source = np.arange(
        5 * 4 * 3,
        dtype=np.int32,
    ).reshape(
        5,
        4,
        3,
    )

    output, metadata = crop_or_pad_xyz(
        source,
        start_xyz=(
            -2,
            0,
            -1,
        ),
        target_shape_xyz=(
            7,
            4,
            4,
        ),
        fill_value=-1,
    )

    assert output.shape == (
        7,
        4,
        4,
    )

    assert (
        metadata.source_start_xyz
        == (
            0,
            0,
            0,
        )
    )

    assert (
        metadata.source_end_xyz_exclusive
        == (
            5,
            4,
            3,
        )
    )

    assert (
        metadata.destination_start_xyz
        == (
            2,
            0,
            1,
        )
    )

    assert (
        metadata.destination_end_xyz_exclusive
        == (
            7,
            4,
            4,
        )
    )

    assert np.array_equal(
        output[
            2:7,
            0:4,
            1:4,
        ],
        source,
    )

    assert np.all(
        output[
            0:2
        ]
        == -1
    )


def test_model_to_canonical_uses_destination_offset():
    source = np.zeros(
        (
            5,
            4,
            3,
        ),
        dtype=np.uint8,
    )

    _, metadata = crop_or_pad_xyz(
        source,
        start_xyz=(
            -2,
            0,
            -1,
        ),
        target_shape_xyz=(
            7,
            4,
            4,
        ),
        fill_value=0,
    )

    canonical = model_xyz_to_canonical_xyz(
        model_xyz=(
            2,
            0,
            1,
        ),
        metadata=metadata,
    )

    assert np.array_equal(
        canonical,
        np.asarray(
            (
                0,
                0,
                0,
            )
        ),
    )

    canonical_last = (
        model_xyz_to_canonical_xyz(
            model_xyz=(
                6,
                3,
                3,
            ),
            metadata=metadata,
        )
    )

    assert np.array_equal(
        canonical_last,
        np.asarray(
            (
                4,
                3,
                2,
            )
        ),
    )


def test_model_padding_coordinate_is_rejected():
    source = np.zeros(
        (
            5,
            4,
            3,
        ),
        dtype=np.uint8,
    )

    _, metadata = crop_or_pad_xyz(
        source,
        start_xyz=(
            -2,
            0,
            -1,
        ),
        target_shape_xyz=(
            7,
            4,
            4,
        ),
        fill_value=0,
    )

    with pytest.raises(
        AdapterIntegrityError,
        match="padded",
    ):
        model_xyz_to_canonical_xyz(
            model_xyz=(
                1,
                0,
                1,
            ),
            metadata=metadata,
        )


def test_canonical_to_model_matches_frozen_start_subtraction():
    model = canonical_xyz_to_model_xyz(
        canonical_xyz=(
            65,
            70,
            30,
        ),
        crop_start_xyz=(
            65,
            63,
            21,
        ),
    )

    assert np.array_equal(
        model,
        np.asarray(
            (
                0,
                7,
                9,
            )
        ),
    )


def test_positive_intensity_normalization_ddof_zero():
    image = np.zeros(
        (
            4,
            4,
            4,
        ),
        dtype=np.float32,
    )

    image[
        1,
        1,
        1,
    ] = 1.0

    image[
        1,
        1,
        2,
    ] = 3.0

    image[
        1,
        2,
        1,
    ] = 5.0

    normalized, mean, std, count = (
        normalize_positive_model_image(
            image
        )
    )

    expected = np.asarray(
        [
            1.0,
            3.0,
            5.0,
        ],
        dtype=np.float64,
    )

    assert count == 3

    assert np.isclose(
        mean,
        expected.mean(),
    )

    assert np.isclose(
        std,
        expected.std(
            ddof=0
        ),
    )

    positive = normalized[
        image > 0
    ].astype(
        np.float64
    )

    assert abs(
        positive.mean()
    ) < 1e-6

    assert abs(
        positive.std(
            ddof=0
        )
        - 1.0
    ) < 1e-6

    assert np.all(
        normalized[
            image <= 0
        ]
        == 0.0
    )


def test_zero_variance_positive_image_is_integrity_failure():
    image = np.zeros(
        (
            4,
            4,
            4,
        ),
        dtype=np.float32,
    )

    image[
        1:3,
        1:3,
        1:3,
    ] = 5.0

    with pytest.raises(
        AdapterIntegrityError,
        match="statistics are invalid",
    ):
        normalize_positive_model_image(
            image
        )


def test_nonintegral_coordinate_transform_is_hard_failure():
    native_affine = np.eye(
        4,
        dtype=np.float64,
    )

    canonical_affine = np.eye(
        4,
        dtype=np.float64,
    )

    # Deliberately create a half-voxel world translation.
    native_affine[
        0,
        3,
    ] = 0.5

    with pytest.raises(
        AdapterIntegrityError,
        match="not integral",
    ):
        native_zyx_to_canonical_xyz(
            (
                1,
                1,
                1,
            ),
            native_affine,
            canonical_affine,
        )
