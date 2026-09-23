from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import src.pipeline.automatic as automatic

from src.inference.baseline import (
    DeterministicBaselineResult,
)
from src.pipeline.policy import (
    ACTION_ABSTAIN_BASELINE,
    ACTION_APPLY_LOCAL_FP,
    GATE_ABSTAIN,
    GATE_FP_ELIGIBLE,
)
from src.sam.adapter import (
    PromptPreservingCropInfeasible,
)
from src.sam.inference import (
    SAMIntegrityError,
    SAMPointBranchResult,
)
from src.training.config import (
    BaselineTrainingConfig,
)
from src.uncertainty.mc_dropout import (
    MCUncertaintyResult,
)


SHAPE_ZYX = (
    24,
    24,
    24,
)


def _raw_mri() -> torch.Tensor:
    z, y, x = np.indices(
        SHAPE_ZYX
    )

    signal = (
        z
        + 2 * y
        + 3 * x
        + 1
    ).astype(
        np.float32
    )

    raw = np.zeros(
        (
            4,
            *SHAPE_ZYX,
        ),
        dtype=np.float32,
    )

    foreground = (
        slice(4, 20),
        slice(4, 20),
        slice(4, 20),
    )

    for channel in range(4):
        raw[
            channel
        ][
            foreground
        ] = (
            signal[
                foreground
            ]
            + float(
                channel
            )
        )

    return torch.from_numpy(
        raw
    )


def _coarse_mask() -> np.ndarray:
    coarse = np.zeros(
        SHAPE_ZYX,
        dtype=bool,
    )

    coarse[
        8:16,
        8:16,
        8:16,
    ] = True

    return coarse


def _write_t2f(
    tmp_path: Path,
    raw_mri: torch.Tensor,
) -> Path:
    t2f_zyx = (
        raw_mri[
            3
        ]
        .numpy()
    )

    t2f_xyz = np.transpose(
        t2f_zyx,
        (
            2,
            1,
            0,
        ),
    ).astype(
        np.float32,
        copy=False,
    )

    path = (
        tmp_path
        / "synthetic-t2f.nii.gz"
    )

    nib.save(
        nib.Nifti1Image(
            t2f_xyz,
            np.eye(
                4,
                dtype=np.float64,
            ),
        ),
        str(
            path
        ),
    )

    return path


def _patch_baseline_and_mc(
    monkeypatch,
    coarse: np.ndarray,
    variance: np.ndarray,
):
    probability = np.where(
        coarse,
        0.9,
        0.1,
    ).astype(
        np.float32
    )

    def fake_baseline(
        model,
        config,
        prepared_mri_czyx,
        *,
        device,
    ):
        return DeterministicBaselineResult(
            probability_zyx=(
                probability.copy()
            ),
            coarse_mask_zyx=(
                coarse.copy()
            ),
        )

    def fake_mc(
        model,
        mri_batch,
        roi_size,
        overlap,
        case_id,
    ):
        return MCUncertaintyResult(
            case_seed=123456,
            activated_dropout_names=(
                "d0",
                "d1",
                "d2",
                "d3",
                "d4",
            ),
            pass_hashes=tuple(
                f"pass-{index}"
                for index in range(
                    10
                )
            ),
            probability_stack=np.zeros(
                (
                    10,
                    *SHAPE_ZYX,
                ),
                dtype=np.float32,
            ),
            mean_probability=np.zeros(
                SHAPE_ZYX,
                dtype=np.float32,
            ),
            predictive_variance=(
                variance
                .astype(
                    np.float32,
                    copy=True,
                )
            ),
        )

    monkeypatch.setattr(
        automatic,
        "run_deterministic_baseline",
        fake_baseline,
    )

    monkeypatch.setattr(
        automatic,
        "run_mc_dropout_case",
        fake_mc,
    )


def _frozen_fake_sam():
    model = torch.nn.Identity()
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    return model


def _patch_sam_execution(
    monkeypatch,
    *,
    behavior: str,
):
    calls = {
        "crop_points": None,
        "prepared": None,
        "image_tensor_calls": 0,
        "encode_calls": 0,
        "sam_points": [],
    }

    real_prepare = (
        automatic
        .prepare_sam_t2f_input
    )

    def capture_prepare(
        t2f_path,
        coarse_mask_zyx,
        crop_prompt_points_zyx,
    ):
        calls[
            "crop_points"
        ] = dict(
            crop_prompt_points_zyx
        )

        result = real_prepare(
            t2f_path=t2f_path,
            coarse_mask_zyx=(
                coarse_mask_zyx
            ),
            crop_prompt_points_zyx=(
                crop_prompt_points_zyx
            ),
        )

        calls[
            "prepared"
        ] = result

        return result

    def fake_prepare_tensor(
        image_model_xyz,
        *,
        device,
    ):
        calls[
            "image_tensor_calls"
        ] += 1

        assert (
            image_model_xyz.shape
            == (
                128,
                128,
                128,
            )
        )

        return torch.zeros(
            (
                1,
            ),
            dtype=torch.float32,
            device=device,
        )

    def fake_encode(
        sam_model,
        image_tensor,
    ):
        calls[
            "encode_calls"
        ] += 1

        return torch.zeros(
            (
                1,
            ),
            dtype=torch.float32,
            device=(
                image_tensor.device
            ),
        )

    def fake_run_branch(
        sam_model,
        image_embedding,
        coordinate_xyz,
        label,
    ):
        coordinate = tuple(
            int(value)
            for value in coordinate_xyz
        )

        calls[
            "sam_points"
        ].append(
            (
                coordinate,
                int(label),
            )
        )

        if behavior == "raise":
            raise SAMIntegrityError(
                "synthetic hard failure"
            )

        prepared = calls[
            "prepared"
        ]

        assert prepared is not None

        mask = np.ones(
            prepared
            .image_model_xyz
            .shape,
            dtype=bool,
        )

        if behavior == "remove":
            mask[
                coordinate
            ] = False

        elif behavior != "keep":
            raise AssertionError(
                f"Unknown behavior: {behavior}"
            )

        probability = (
            mask.astype(
                np.float32
            )
        )

        return SAMPointBranchResult(
            coordinate_xyz=(
                coordinate
            ),
            label=int(
                label
            ),
            probability_model_xyz=(
                probability
            ),
            mask_model_xyz=(
                mask
            ),
        )

    monkeypatch.setattr(
        automatic,
        "prepare_sam_t2f_input",
        capture_prepare,
    )

    monkeypatch.setattr(
        automatic,
        "prepare_sam_image_tensor",
        fake_prepare_tensor,
    )

    monkeypatch.setattr(
        automatic,
        "encode_sam_image",
        fake_encode,
    )

    monkeypatch.setattr(
        automatic,
        "run_sam_point_branch",
        fake_run_branch,
    )

    return calls


def _run(
    *,
    raw_mri,
    t2f_path=None,
    sam_model=None,
):
    return (
        automatic
        .run_final_automatic_pipeline(
            baseline_model=(
                torch.nn.Identity()
                .eval()
            ),
            baseline_config=(
                BaselineTrainingConfig()
            ),
            raw_mri_czyx=(
                raw_mri
            ),
            case_id=(
                "BraTS-SYNTHETIC-001"
            ),
            spacing_zyx_mm=(
                1.0,
                1.0,
                1.0,
            ),
            device="cpu",
            t2f_path=t2f_path,
            sam_model=sam_model,
        )
    )


def test_variance_gate_abstain_returns_exact_baseline_and_skips_prompt_sam(
    monkeypatch,
):
    raw = _raw_mri()
    coarse = _coarse_mask()

    variance = np.zeros(
        SHAPE_ZYX,
        dtype=np.float32,
    )

    _patch_baseline_and_mc(
        monkeypatch,
        coarse,
        variance,
    )

    def forbidden(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Downstream refinement must not run for gate abstention."
        )

    monkeypatch.setattr(
        automatic,
        "generate_autoprompt_v2",
        forbidden,
    )

    monkeypatch.setattr(
        automatic,
        "prepare_sam_t2f_input",
        forbidden,
    )

    result = _run(
        raw_mri=raw
    )

    assert (
        result.action
        == ACTION_ABSTAIN_BASELINE
    )

    assert (
        result.gate_state
        == GATE_ABSTAIN
    )

    assert (
        result.semantic_abstention_condition
        is None
    )

    assert result.sam_used is False

    assert np.array_equal(
        result.final_mask_zyx,
        coarse,
    )

    assert not np.any(
        result.removal_mask_zyx
    )

    assert not np.any(
        result.support_mask_zyx
    )


def test_empty_mri_foreground_semantically_abstains_before_prompts(
    monkeypatch,
):
    raw = torch.zeros(
        (
            4,
            *SHAPE_ZYX,
        ),
        dtype=torch.float32,
    )

    coarse = _coarse_mask()

    variance = np.ones(
        SHAPE_ZYX,
        dtype=np.float32,
    )

    _patch_baseline_and_mc(
        monkeypatch,
        coarse,
        variance,
    )

    def forbidden(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Prompt/SAM path must not run after semantic abstention."
        )

    monkeypatch.setattr(
        automatic,
        "generate_autoprompt_v2",
        forbidden,
    )

    result = _run(
        raw_mri=raw
    )

    assert (
        result.action
        == ACTION_ABSTAIN_BASELINE
    )

    assert (
        result.semantic_abstention_condition
        == "empty MRI foreground"
    )

    assert result.gate_state is None
    assert result.sam_used is False

    assert np.array_equal(
        result.final_mask_zyx,
        coarse,
    )


def test_missing_fp_hypothesis_semantically_abstains_and_skips_sam(
    monkeypatch,
):
    coarse = _coarse_mask()

    raw_array = np.zeros(
        (
            4,
            *SHAPE_ZYX,
        ),
        dtype=np.float32,
    )

    raw_array[
        :,
        7,
        12,
        12,
    ] = np.asarray(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        dtype=np.float32,
    )[
        :,
        None,
    ].reshape(
        4,
    )

    raw = torch.from_numpy(
        raw_array
    )

    variance = np.zeros(
        SHAPE_ZYX,
        dtype=np.float32,
    )

    variance[
        7,
        12,
        12,
    ] = 0.5

    _patch_baseline_and_mc(
        monkeypatch,
        coarse,
        variance,
    )

    def forbidden(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "SAM preparation must not run without an FP hypothesis."
        )

    monkeypatch.setattr(
        automatic,
        "prepare_sam_t2f_input",
        forbidden,
    )

    result = _run(
        raw_mri=raw
    )

    assert (
        result.action
        == ACTION_ABSTAIN_BASELINE
    )

    assert (
        result.gate_state
        == GATE_FP_ELIGIBLE
    )

    assert (
        result.semantic_abstention_condition
        == "no valid FP correction hypothesis"
    )

    assert result.sam_used is False

    assert np.array_equal(
        result.final_mask_zyx,
        coarse,
    )


def test_crop_infeasible_is_semantic_abstention_not_hard_failure(
    monkeypatch,
):
    raw = _raw_mri()
    coarse = _coarse_mask()

    variance = np.zeros(
        SHAPE_ZYX,
        dtype=np.float32,
    )

    variance[
        8,
        12,
        12,
    ] = 0.5

    _patch_baseline_and_mc(
        monkeypatch,
        coarse,
        variance,
    )

    def infeasible(
        *args,
        **kwargs,
    ):
        raise PromptPreservingCropInfeasible(
            "synthetic infeasible crop"
        )

    monkeypatch.setattr(
        automatic,
        "prepare_sam_t2f_input",
        infeasible,
    )

    result = _run(
        raw_mri=raw,
        t2f_path="unused.nii.gz",
    )

    assert (
        result.action
        == ACTION_ABSTAIN_BASELINE
    )

    assert (
        result.gate_state
        == GATE_FP_ELIGIBLE
    )

    assert (
        result.semantic_abstention_condition
        == (
            "frozen prompt-preserving "
            "crop geometry infeasible"
        )
    )

    assert result.sam_used is False

    assert np.array_equal(
        result.final_mask_zyx,
        coarse,
    )


def test_fp_eligible_executes_one_negative_sam_point_and_removal_only(
    tmp_path,
    monkeypatch,
):
    raw = _raw_mri()
    coarse = _coarse_mask()

    variance = np.zeros(
        SHAPE_ZYX,
        dtype=np.float32,
    )

    variance[
        8,
        12,
        12,
    ] = 0.5

    _patch_baseline_and_mc(
        monkeypatch,
        coarse,
        variance,
    )

    calls = _patch_sam_execution(
        monkeypatch,
        behavior="remove",
    )

    t2f_path = _write_t2f(
        tmp_path,
        raw,
    )

    result = _run(
        raw_mri=raw,
        t2f_path=t2f_path,
        sam_model=(
            _frozen_fake_sam()
        ),
    )

    assert (
        result.action
        == ACTION_APPLY_LOCAL_FP
    )

    assert (
        result.gate_state
        == GATE_FP_ELIGIBLE
    )

    assert (
        result.semantic_abstention_condition
        is None
    )

    assert result.sam_used is True

    assert set(
        calls[
            "crop_points"
        ]
    ) == {
        "uncertainty_hotspot",
        "false_positive_correction",
        "false_negative_correction",
    }

    assert not any(
        "control"
        in name.lower()
        for name in calls[
            "crop_points"
        ]
    )

    assert len(
        calls[
            "sam_points"
        ]
    ) == 1

    (
        sam_coordinate,
        sam_label,
    ) = calls[
        "sam_points"
    ][
        0
    ]

    assert sam_label == 0

    assert (
        sam_coordinate
        == result.fp_prompt_model_xyz
    )

    assert (
        int(
            result
            .removal_mask_zyx
            .sum()
        )
        == 1
    )

    assert np.all(
        np.logical_not(
            result.final_mask_zyx
        )
        | result.baseline_mask_zyx
    )

    assert np.all(
        np.logical_not(
            result.removal_mask_zyx
        )
        | result.support_mask_zyx
    )

    assert np.array_equal(
        result.final_mask_zyx[
            ~result.support_mask_zyx
        ],
        result.baseline_mask_zyx[
            ~result.support_mask_zyx
        ],
    )


def test_fp_eligible_zero_edit_is_valid_and_returns_unchanged_baseline(
    tmp_path,
    monkeypatch,
):
    raw = _raw_mri()
    coarse = _coarse_mask()

    variance = np.zeros(
        SHAPE_ZYX,
        dtype=np.float32,
    )

    variance[
        8,
        12,
        12,
    ] = 0.5

    _patch_baseline_and_mc(
        monkeypatch,
        coarse,
        variance,
    )

    _patch_sam_execution(
        monkeypatch,
        behavior="keep",
    )

    t2f_path = _write_t2f(
        tmp_path,
        raw,
    )

    result = _run(
        raw_mri=raw,
        t2f_path=t2f_path,
        sam_model=(
            _frozen_fake_sam()
        ),
    )

    assert (
        result.action
        == ACTION_APPLY_LOCAL_FP
    )

    assert result.sam_used is True

    assert (
        int(
            result
            .removal_mask_zyx
            .sum()
        )
        == 0
    )

    assert np.array_equal(
        result.final_mask_zyx,
        coarse,
    )


def test_missing_fn_hypothesis_does_not_block_final_fp_branch(
    tmp_path,
    monkeypatch,
):
    coarse = _coarse_mask()

    raw = _raw_mri()

    raw_array = (
        raw
        .numpy()
        .copy()
    )

    raw_array[
        :,
        ~coarse,
    ] = 0.0

    raw = torch.from_numpy(
        raw_array
    )

    variance = np.zeros(
        SHAPE_ZYX,
        dtype=np.float32,
    )

    variance[
        8,
        12,
        12,
    ] = 0.5

    _patch_baseline_and_mc(
        monkeypatch,
        coarse,
        variance,
    )

    calls = _patch_sam_execution(
        monkeypatch,
        behavior="keep",
    )

    t2f_path = _write_t2f(
        tmp_path,
        raw,
    )

    result = _run(
        raw_mri=raw,
        t2f_path=t2f_path,
        sam_model=(
            _frozen_fake_sam()
        ),
    )

    assert (
        result.action
        == ACTION_APPLY_LOCAL_FP
    )

    assert result.sam_used is True

    assert (
        "false_negative_correction"
        not in calls[
            "crop_points"
        ]
    )

    assert (
        "false_positive_correction"
        in calls[
            "crop_points"
        ]
    )

    assert len(
        calls[
            "sam_points"
        ]
    ) == 1

    assert (
        calls[
            "sam_points"
        ][
            0
        ][
            1
        ]
        == 0
    )


def test_sam_integrity_failure_propagates_as_hard_failure(
    tmp_path,
    monkeypatch,
):
    raw = _raw_mri()
    coarse = _coarse_mask()

    variance = np.zeros(
        SHAPE_ZYX,
        dtype=np.float32,
    )

    variance[
        8,
        12,
        12,
    ] = 0.5

    _patch_baseline_and_mc(
        monkeypatch,
        coarse,
        variance,
    )

    _patch_sam_execution(
        monkeypatch,
        behavior="raise",
    )

    t2f_path = _write_t2f(
        tmp_path,
        raw,
    )

    with pytest.raises(
        SAMIntegrityError,
        match="synthetic hard failure",
    ):
        _run(
            raw_mri=raw,
            t2f_path=t2f_path,
            sam_model=(
                _frozen_fake_sam()
            ),
        )
