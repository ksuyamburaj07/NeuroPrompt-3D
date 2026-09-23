import numpy as np
import pytest

from src.prompts.automatic import (
    FINAL_V1_SELECTED_BRANCH,
    FN_BRANCH,
    FN_LABEL,
    FP_BRANCH,
    FP_LABEL,
    NO_VALID_FP_CONDITION,
    AutoPromptV2Result,
    PromptCandidate,
    PromptIntegrityError,
    generate_autoprompt_v2,
    select_final_v1_fp_prompt,
)


def _base_inputs():
    mri = np.ones(
        (
            4,
            11,
            11,
            11,
        ),
        dtype=np.float32,
    )

    coarse = np.zeros(
        (
            11,
            11,
            11,
        ),
        dtype=bool,
    )

    coarse[
        3:8,
        3:8,
        3:8,
    ] = True

    return (
        mri,
        coarse,
    )


def test_frozen_prompt_labels_and_final_branch():
    assert FP_LABEL == 0
    assert FN_LABEL == 1
    assert FINAL_V1_SELECTED_BRANCH == FP_BRANCH


def test_generate_v2_from_hotspot_inside_coarse():
    mri, coarse = (
        _base_inputs()
    )

    hotspot = (
        3,
        5,
        5,
    )

    result = generate_autoprompt_v2(
        mri,
        coarse,
        hotspot,
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    assert (
        result.fp_candidate
        is not None
    )

    assert (
        result.fn_candidate
        is not None
    )

    # Hotspot itself is a valid FP candidate.
    assert (
        result.fp_candidate.coordinate_zyx
        == hotspot
    )

    assert result.fp_candidate.label == 0

    # Nearest exterior voxel is one voxel away.
    assert np.isclose(
        result.fn_candidate.distance_mm,
        1.0,
    )

    assert result.fn_candidate.label == 1


def test_generate_v2_from_hotspot_outside_coarse():
    mri, coarse = (
        _base_inputs()
    )

    hotspot = (
        2,
        5,
        5,
    )

    result = generate_autoprompt_v2(
        mri,
        coarse,
        hotspot,
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    # Hotspot itself is the valid FN candidate.
    assert (
        result.fn_candidate.coordinate_zyx
        == hotspot
    )

    assert result.fn_candidate.label == 1

    assert (
        result.fp_candidate.coordinate_zyx
        == (
            3,
            5,
            5,
        )
    )

    assert result.fp_candidate.label == 0


def test_nearest_candidate_uses_physical_mm():
    mri = np.zeros(
        (
            4,
            9,
            9,
            9,
        ),
        dtype=np.float32,
    )

    coarse = np.zeros(
        (
            9,
            9,
            9,
        ),
        dtype=bool,
    )

    hotspot = (
        4,
        4,
        4,
    )

    # Hotspot must itself be MRI foreground.
    mri[
        :,
        hotspot[0],
        hotspot[1],
        hotspot[2],
    ] = 1.0

    # FP candidate A:
    # 1 voxel in Z, but Z spacing is 5 mm.
    candidate_a = (
        5,
        4,
        4,
    )

    # FP candidate B:
    # 2 voxels in X, X spacing is 1 mm = 2 mm.
    candidate_b = (
        4,
        4,
        6,
    )

    for point in [
        candidate_a,
        candidate_b,
    ]:
        mri[
            :,
            point[0],
            point[1],
            point[2],
        ] = 1.0

        coarse[
            point
        ] = True

    result = generate_autoprompt_v2(
        mri,
        coarse,
        hotspot,
        spacing_zyx_mm=(
            5.0,
            2.0,
            1.0,
        ),
    )

    assert (
        result.fp_candidate.coordinate_zyx
        == candidate_b
    )

    assert np.isclose(
        result.fp_candidate.distance_mm,
        2.0,
    )


def test_exact_distance_tie_uses_first_argwhere_coordinate():
    mri = np.zeros(
        (
            4,
            9,
            9,
            9,
        ),
        dtype=np.float32,
    )

    coarse = np.zeros(
        (
            9,
            9,
            9,
        ),
        dtype=bool,
    )

    hotspot = (
        4,
        4,
        4,
    )

    mri[
        :,
        hotspot[0],
        hotspot[1],
        hotspot[2],
    ] = 1.0

    first = (
        4,
        3,
        4,
    )

    second = (
        4,
        5,
        4,
    )

    for point in [
        first,
        second,
    ]:
        mri[
            :,
            point[0],
            point[1],
            point[2],
        ] = 1.0

        coarse[
            point
        ] = True

    result = generate_autoprompt_v2(
        mri,
        coarse,
        hotspot,
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    assert (
        result.fp_candidate.coordinate_zyx
        == first
    )


def test_candidate_must_be_in_mri_foreground():
    mri = np.zeros(
        (
            4,
            9,
            9,
            9,
        ),
        dtype=np.float32,
    )

    coarse = np.zeros(
        (
            9,
            9,
            9,
        ),
        dtype=bool,
    )

    hotspot = (
        4,
        4,
        4,
    )

    # Hotspot is MRI foreground but outside coarse.
    mri[
        :,
        hotspot[0],
        hotspot[1],
        hotspot[2],
    ] = 1.0

    # Very close coarse voxel is NOT MRI foreground.
    coarse[
        4,
        4,
        5,
    ] = True

    # Farther coarse voxel is MRI foreground and therefore valid.
    valid_fp = (
        4,
        4,
        7,
    )

    coarse[
        valid_fp
    ] = True

    mri[
        :,
        valid_fp[0],
        valid_fp[1],
        valid_fp[2],
    ] = 1.0

    result = generate_autoprompt_v2(
        mri,
        coarse,
        hotspot,
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    assert (
        result.fp_candidate.coordinate_zyx
        == valid_fp
    )


def test_missing_fp_candidate_is_final_semantic_abstention():
    mri = np.ones(
        (
            4,
            7,
            7,
            7,
        ),
        dtype=np.float32,
    )

    coarse = np.zeros(
        (
            7,
            7,
            7,
        ),
        dtype=bool,
    )

    result = generate_autoprompt_v2(
        mri,
        coarse,
        hotspot_zyx=(
            3,
            3,
            3,
        ),
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    assert result.fp_candidate is None
    assert result.fn_candidate is not None

    selection = (
        select_final_v1_fp_prompt(
            result
        )
    )

    assert not selection.available

    assert (
        selection.abstention_condition
        == NO_VALID_FP_CONDITION
    )


def test_final_v1_selects_exactly_fp_negative_branch():
    fp = PromptCandidate(
        branch=FP_BRANCH,
        coordinate_zyx=(
            4,
            5,
            6,
        ),
        label=FP_LABEL,
        distance_mm=2.0,
    )

    fn = PromptCandidate(
        branch=FN_BRANCH,
        coordinate_zyx=(
            4,
            5,
            7,
        ),
        label=FN_LABEL,
        distance_mm=3.0,
    )

    result = AutoPromptV2Result(
        hotspot_zyx=(
            4,
            5,
            5,
        ),
        fp_candidate=fp,
        fn_candidate=fn,
    )

    selection = (
        select_final_v1_fp_prompt(
            result
        )
    )

    assert selection.available
    assert selection.candidate == fp

    assert (
        selection.candidate.branch
        == FP_BRANCH
    )

    assert (
        selection.candidate.label
        == 0
    )

    assert selection.candidate != fn


def test_final_v1_rejects_malformed_fp_candidate():
    malformed = PromptCandidate(
        branch=FN_BRANCH,
        coordinate_zyx=(
            1,
            1,
            1,
        ),
        label=FN_LABEL,
        distance_mm=1.0,
    )

    result = AutoPromptV2Result(
        hotspot_zyx=(
            1,
            1,
            1,
        ),
        fp_candidate=malformed,
        fn_candidate=None,
    )

    with pytest.raises(
        PromptIntegrityError,
        match="violates the frozen prompt contract",
    ):
        select_final_v1_fp_prompt(
            result
        )


def test_hotspot_outside_geometry_is_integrity_failure():
    mri, coarse = (
        _base_inputs()
    )

    with pytest.raises(
        PromptIntegrityError,
        match="outside MRI geometry",
    ):
        generate_autoprompt_v2(
            mri,
            coarse,
            hotspot_zyx=(
                99,
                5,
                5,
            ),
            spacing_zyx_mm=(
                1.0,
                1.0,
                1.0,
            ),
        )


def test_hotspot_not_in_mri_foreground_is_integrity_failure():
    mri, coarse = (
        _base_inputs()
    )

    hotspot = (
        3,
        5,
        5,
    )

    mri[
        :,
        hotspot[0],
        hotspot[1],
        hotspot[2],
    ] = 0.0

    with pytest.raises(
        PromptIntegrityError,
        match="must lie in MRI foreground",
    ):
        generate_autoprompt_v2(
            mri,
            coarse,
            hotspot,
            spacing_zyx_mm=(
                1.0,
                1.0,
                1.0,
            ),
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
def test_invalid_spacing_is_integrity_failure(spacing):
    mri, coarse = (
        _base_inputs()
    )

    with pytest.raises(
        PromptIntegrityError,
    ):
        generate_autoprompt_v2(
            mri,
            coarse,
            hotspot_zyx=(
                3,
                5,
                5,
            ),
            spacing_zyx_mm=spacing,
        )
