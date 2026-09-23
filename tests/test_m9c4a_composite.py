import numpy as np
import pytest

from src.pipeline.composite import (
    BOUNDARY_SHELL_MM,
    LOCAL_RADIUS_MM,
    CompositeIntegrityError,
    apply_local_fp_composite,
    embed_model_mask_full,
    fov_coverage_mask,
    physical_ball_mask,
    physical_boundary_shell,
)


def test_frozen_physical_constants():
    assert LOCAL_RADIUS_MM == 15.0
    assert BOUNDARY_SHELL_MM == 10.0


def test_anisotropic_physical_ball_uses_mm_not_voxel_distance():
    ball = physical_ball_mask(
        shape_xyz=(9, 9, 9),
        center_xyz=(4, 4, 4),
        spacing_xyz_mm=(5.0, 2.0, 1.0),
        radius_mm=4.0,
    )

    # 5 mm away in X: outside.
    assert not ball[
        5,
        4,
        4,
    ]

    # 4 mm away in Y: inclusive boundary.
    assert ball[
        4,
        6,
        4,
    ]

    # 4 mm away in Z: inclusive boundary.
    assert ball[
        4,
        4,
        8,
    ]

    # sqrt(2^2 + 3^2) < 4 mm.
    assert ball[
        4,
        5,
        7,
    ]


def test_physical_boundary_shell_is_two_sided_and_anisotropic():
    coarse = np.zeros(
        (7, 7, 7),
        dtype=bool,
    )

    coarse[
        1:6,
        1:6,
        1:6,
    ] = True

    shell = physical_boundary_shell(
        coarse,
        spacing_xyz_mm=(3.0, 2.0, 1.0),
        shell_mm=2.0,
    )

    # Inside boundary voxel.
    assert shell[
        3,
        3,
        1,
    ]

    # Outside voxel 1 mm from the Z face.
    assert shell[
        3,
        3,
        0,
    ]

    # Deep interior is farther than 2 mm.
    assert not shell[
        3,
        3,
        3,
    ]

    # Outside X face is 3 mm away, so excluded.
    assert not shell[
        0,
        3,
        3,
    ]


def test_leading_padding_embedding_uses_destination_offset():
    # Synthetic analogue of the real M9B1 leading-padding case:
    # model positions before destination_start are padding and must
    # never be copied into the full source volume.
    crop_policy = {
        "target_shape_xyz": [
            7,
            6,
            5,
        ],
        "source_start_xyz": [
            0,
            1,
            0,
        ],
        "source_end_xyz_exclusive": [
            5,
            5,
            4,
        ],
        "destination_start_xyz": [
            2,
            0,
            1,
        ],
        "destination_end_xyz_exclusive": [
            7,
            4,
            5,
        ],
    }

    model = np.zeros(
        (7, 6, 5),
        dtype=bool,
    )

    # This is padded model space and must disappear.
    model[
        0,
        0,
        0,
    ] = True

    # First valid destination voxel maps to first source voxel.
    model[
        2,
        0,
        1,
    ] = True

    # Last valid destination voxel maps to last source voxel.
    model[
        6,
        3,
        4,
    ] = True

    embedded = embed_model_mask_full(
        model,
        full_shape_xyz=(8, 7, 6),
        crop_policy=crop_policy,
    )

    assert int(
        embedded.sum()
    ) == 2

    assert embedded[
        0,
        1,
        0,
    ]

    assert embedded[
        4,
        4,
        3,
    ]


def test_fov_coverage_is_exact_source_region():
    crop_policy = {
        "target_shape_xyz": [
            7,
            6,
            5,
        ],
        "source_start_xyz": [
            0,
            1,
            0,
        ],
        "source_end_xyz_exclusive": [
            5,
            5,
            4,
        ],
        "destination_start_xyz": [
            2,
            0,
            1,
        ],
        "destination_end_xyz_exclusive": [
            7,
            4,
            5,
        ],
    }

    fov = fov_coverage_mask(
        full_shape_xyz=(8, 7, 6),
        crop_policy=crop_policy,
    )

    expected = np.zeros(
        (8, 7, 6),
        dtype=bool,
    )

    expected[
        0:5,
        1:5,
        0:4,
    ] = True

    assert np.array_equal(
        fov,
        expected,
    )


def test_fp_composite_is_removal_only_and_localized():
    shape = (
        45,
        45,
        45,
    )

    coarse = np.zeros(
        shape,
        dtype=bool,
    )

    coarse[
        5:40,
        5:40,
        5:40,
    ] = True

    sam = (
        coarse.copy()
    )

    fp_point = (
        5,
        22,
        22,
    )

    # Valid local removals.
    sam[
        5,
        22,
        22,
    ] = False

    sam[
        6,
        22,
        22,
    ] = False

    # Deep/distant disagreement: outside 15 mm support.
    sam[
        25,
        22,
        22,
    ] = False

    # SAM foreground in coarse background must never be added.
    sam[
        4,
        22,
        22,
    ] = True

    fov = np.ones(
        shape,
        dtype=bool,
    )

    result, removal, support = (
        apply_local_fp_composite(
            coarse,
            sam,
            fov,
            fp_point,
            spacing_xyz_mm=(
                1.0,
                1.0,
                1.0,
            ),
        )
    )

    assert removal[
        5,
        22,
        22,
    ]

    assert removal[
        6,
        22,
        22,
    ]

    assert not removal[
        25,
        22,
        22,
    ]

    assert result[
        25,
        22,
        22,
    ]

    assert not result[
        4,
        22,
        22,
    ]

    changed = (
        result
        != coarse
    )

    # Every edit is foreground -> background.
    assert np.all(
        coarse[
            changed
        ]
    )

    assert np.all(
        np.logical_not(
            result[
                changed
            ]
        )
    )

    # No modification outside frozen support.
    assert not np.any(
        changed
        & np.logical_not(
            support
        )
    )


def test_fp_composite_respects_valid_sam_fov():
    shape = (
        40,
        40,
        40,
    )

    coarse = np.zeros(
        shape,
        dtype=bool,
    )

    coarse[
        5:35,
        5:35,
        5:35,
    ] = True

    sam = (
        coarse.copy()
    )

    fp_point = (
        5,
        20,
        20,
    )

    sam[
        5,
        20,
        20,
    ] = False

    sam[
        6,
        20,
        20,
    ] = False

    fov = np.ones(
        shape,
        dtype=bool,
    )

    # Exclude one otherwise-valid edit from SAM coverage.
    fov[
        6,
        20,
        20,
    ] = False

    result, removal, _ = (
        apply_local_fp_composite(
            coarse,
            sam,
            fov,
            fp_point,
            spacing_xyz_mm=(
                1.0,
                1.0,
                1.0,
            ),
        )
    )

    assert removal[
        5,
        20,
        20,
    ]

    assert not removal[
        6,
        20,
        20,
    ]

    assert not result[
        5,
        20,
        20,
    ]

    assert result[
        6,
        20,
        20,
    ]


def test_zero_edit_is_valid_and_returns_equal_baseline():
    shape = (
        33,
        33,
        33,
    )

    coarse = np.zeros(
        shape,
        dtype=bool,
    )

    coarse[
        3:28,
        3:28,
        3:28,
    ] = True

    sam = np.ones(
        shape,
        dtype=bool,
    )

    fov = np.ones(
        shape,
        dtype=bool,
    )

    result, removal, _ = (
        apply_local_fp_composite(
            coarse,
            sam,
            fov,
            fp_point_xyz=(
                3,
                16,
                16,
            ),
            spacing_xyz_mm=(
                1.0,
                1.0,
                1.0,
            ),
        )
    )

    assert int(
        removal.sum()
    ) == 0

    assert np.array_equal(
        result,
        coarse,
    )

    # A copy is returned, not the same mutable object.
    assert result is not coarse


def test_malformed_crop_extent_is_hard_failure():
    crop_policy = {
        "target_shape_xyz": [
            8,
            8,
            8,
        ],
        "source_start_xyz": [
            0,
            0,
            0,
        ],
        "source_end_xyz_exclusive": [
            5,
            5,
            5,
        ],
        "destination_start_xyz": [
            1,
            0,
            0,
        ],
        # X extent is 6 rather than source extent 5.
        "destination_end_xyz_exclusive": [
            7,
            5,
            5,
        ],
    }

    with pytest.raises(
        CompositeIntegrityError,
        match="extents must match",
    ):
        fov_coverage_mask(
            full_shape_xyz=(
                10,
                10,
                10,
            ),
            crop_policy=crop_policy,
        )


@pytest.mark.parametrize(
    "spacing",
    [
        (
            0.0,
            1.0,
            1.0,
        ),
        (
            -1.0,
            1.0,
            1.0,
        ),
        (
            np.nan,
            1.0,
            1.0,
        ),
    ],
)
def test_invalid_physical_spacing_is_hard_failure(spacing):
    with pytest.raises(
        CompositeIntegrityError,
    ):
        physical_ball_mask(
            shape_xyz=(
                9,
                9,
                9,
            ),
            center_xyz=(
                4,
                4,
                4,
            ),
            spacing_xyz_mm=spacing,
        )


def test_shape_mismatch_is_hard_failure():
    coarse = np.zeros(
        (20, 20, 20),
        dtype=bool,
    )

    sam = np.zeros(
        (19, 20, 20),
        dtype=bool,
    )

    fov = np.ones(
        (20, 20, 20),
        dtype=bool,
    )

    with pytest.raises(
        CompositeIntegrityError,
        match="identical 3D shapes",
    ):
        apply_local_fp_composite(
            coarse,
            sam,
            fov,
            fp_point_xyz=(
                5,
                5,
                5,
            ),
            spacing_xyz_mm=(
                1.0,
                1.0,
                1.0,
            ),
        )
