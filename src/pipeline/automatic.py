"""Frozen final automatic NeuroPrompt-3D inference orchestrator.

This module composes the already-frozen M9C4A primitives.

Scientific contract:
- raw four-modality MRI only; no ground truth input;
- deterministic 3D U-Net baseline first;
- frozen T=10 selective MC-Dropout;
- uncertainty hotspot in raw MRI foreground / 10 mm coarse shell;
- strict frozen variance gate;
- semantic failures return the exact deterministic baseline;
- eligible refinement uses exactly one FP negative SAM point;
- FN may participate in prompt-preserving crop geometry but is never
  submitted to SAM for final-v1 refinement;
- SAM output is used only for localized foreground removal;
- no foreground addition and no morphology.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from src.data.preprocessing import (
    normalize_multimodal_foreground_zscore,
)
from src.inference.baseline import (
    DeterministicBaselineResult,
    run_deterministic_baseline,
)
from src.pipeline.composite import (
    apply_local_fp_composite,
    embed_model_mask_full,
    fov_coverage_mask,
)
from src.pipeline.policy import (
    ACTION_ABSTAIN_BASELINE,
    ACTION_APPLY_LOCAL_FP,
    GATE_ABSTAIN,
    GATE_FP_ELIGIBLE,
    final_action_for_gate,
    semantic_abstention_action,
    variance_gate,
)
from src.prompts.automatic import (
    generate_autoprompt_v2,
    select_final_v1_fp_prompt,
)
from src.sam.adapter import (
    CropPadMetadata,
    PromptPreservingCropInfeasible,
)
from src.sam.full_volume import (
    build_canonical_full_volume_geometry,
    canonical_mask_xyz_to_internal_zyx,
)
from src.sam.inference import (
    SAM_NEGATIVE_LABEL,
    encode_sam_image,
    prepare_sam_image_tensor,
    run_sam_point_branch,
)
from src.sam.preprocessing import (
    PreparedSAMInput,
    prepare_sam_t2f_input,
)
from src.training.config import (
    BaselineTrainingConfig,
)
from src.uncertainty.hotspot import (
    HotspotResult,
    select_uncertainty_hotspot,
)
from src.uncertainty.mc_dropout import (
    MCUncertaintyResult,
    run_mc_dropout_case,
)


SEMANTIC_CROP_INFEASIBLE = (
    "frozen prompt-preserving crop geometry infeasible"
)


class AutomaticPipelineIntegrityError(ValueError):
    """Hard failure for a malformed frozen automatic inference contract."""


@dataclass(frozen=True)
class AutomaticPipelineResult:
    """Final-v1 automatic inference result."""

    action: str
    semantic_abstention_condition: Optional[str]
    gate_state: Optional[str]

    case_seed: int
    mc_pass_hashes: tuple[str, ...]

    hotspot_zyx: Optional[tuple[int, int, int]]
    hotspot_variance: Optional[float]

    fp_prompt_zyx: Optional[tuple[int, int, int]]
    fp_prompt_model_xyz: Optional[tuple[int, int, int]]

    sam_used: bool

    baseline_probability_zyx: np.ndarray
    baseline_mask_zyx: np.ndarray
    final_mask_zyx: np.ndarray

    removal_mask_zyx: np.ndarray
    support_mask_zyx: np.ndarray


def _spacing3(
    spacing_zyx_mm: Sequence[float],
) -> np.ndarray:
    spacing = np.asarray(
        spacing_zyx_mm,
        dtype=np.float64,
    )

    if spacing.shape != (3,):
        raise AutomaticPipelineIntegrityError(
            "spacing_zyx_mm must contain exactly three values."
        )

    if (
        not np.isfinite(spacing).all()
        or np.any(spacing <= 0.0)
    ):
        raise AutomaticPipelineIntegrityError(
            "spacing_zyx_mm must contain finite positive values."
        )

    return spacing


def _validate_raw_mri(
    raw_mri_czyx: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(
        raw_mri_czyx,
        torch.Tensor,
    ):
        raise AutomaticPipelineIntegrityError(
            "raw_mri_czyx must be a torch.Tensor."
        )

    if (
        raw_mri_czyx.ndim != 4
        or int(raw_mri_czyx.shape[0]) != 4
    ):
        raise AutomaticPipelineIntegrityError(
            "raw_mri_czyx must have shape [4,D,H,W]."
        )

    if raw_mri_czyx.dtype != torch.float32:
        raise AutomaticPipelineIntegrityError(
            "raw_mri_czyx must be float32."
        )

    if not torch.isfinite(
        raw_mri_czyx
    ).all():
        raise AutomaticPipelineIntegrityError(
            "raw_mri_czyx contains non-finite values."
        )

    return (
        raw_mri_czyx
        .detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )


def _validate_baseline_result(
    result: DeterministicBaselineResult,
    spatial_shape: tuple[int, int, int],
) -> None:
    probability = np.asarray(
        result.probability_zyx
    )

    coarse = np.asarray(
        result.coarse_mask_zyx
    )

    if (
        probability.shape != spatial_shape
        or coarse.shape != spatial_shape
    ):
        raise AutomaticPipelineIntegrityError(
            "Baseline result geometry does not match MRI."
        )

    if not np.isfinite(
        probability
    ).all():
        raise AutomaticPipelineIntegrityError(
            "Baseline probability contains non-finite values."
        )


def _crop_policy_mapping(
    metadata: CropPadMetadata,
) -> dict[str, tuple[int, int, int]]:
    return {
        "target_shape_xyz": tuple(
            int(value)
            for value in metadata.target_shape_xyz
        ),
        "source_start_xyz": tuple(
            int(value)
            for value in metadata.source_start_xyz
        ),
        "source_end_xyz_exclusive": tuple(
            int(value)
            for value in metadata.source_end_xyz_exclusive
        ),
        "destination_start_xyz": tuple(
            int(value)
            for value in metadata.destination_start_xyz
        ),
        "destination_end_xyz_exclusive": tuple(
            int(value)
            for value in metadata.destination_end_xyz_exclusive
        ),
    }


def _native_spacing_zyx_mm(
    prepared: PreparedSAMInput,
) -> np.ndarray:
    affine = np.asarray(
        prepared.native_affine,
        dtype=np.float64,
    )

    if (
        affine.shape != (4, 4)
        or not np.isfinite(affine).all()
    ):
        raise AutomaticPipelineIntegrityError(
            "Prepared SAM native affine is malformed."
        )

    spacing_xyz = np.sqrt(
        np.sum(
            affine[:3, :3] ** 2,
            axis=0,
        )
    )

    spacing_zyx = spacing_xyz[
        ::-1
    ]

    if (
        not np.isfinite(spacing_zyx).all()
        or np.any(spacing_zyx <= 0.0)
    ):
        raise AutomaticPipelineIntegrityError(
            "Prepared SAM native voxel spacing is invalid."
        )

    return spacing_zyx


def _baseline_result(
    *,
    baseline: DeterministicBaselineResult,
    mc: MCUncertaintyResult,
    hotspot: HotspotResult,
    gate_state: Optional[str],
    semantic_condition: Optional[str],
) -> AutomaticPipelineResult:
    if semantic_condition is not None:
        action = semantic_abstention_action(
            semantic_condition
        )

        if action != ACTION_ABSTAIN_BASELINE:
            raise AutomaticPipelineIntegrityError(
                "Semantic abstention did not map to baseline action."
            )

    else:
        action = ACTION_ABSTAIN_BASELINE

    coarse = np.ascontiguousarray(
        baseline.coarse_mask_zyx,
        dtype=bool,
    )

    probability = np.ascontiguousarray(
        baseline.probability_zyx,
        dtype=np.float32,
    )

    zeros = np.zeros_like(
        coarse,
        dtype=bool,
    )

    return AutomaticPipelineResult(
        action=action,
        semantic_abstention_condition=(
            semantic_condition
        ),
        gate_state=gate_state,
        case_seed=int(
            mc.case_seed
        ),
        mc_pass_hashes=tuple(
            mc.pass_hashes
        ),
        hotspot_zyx=(
            hotspot.coordinate_zyx
        ),
        hotspot_variance=(
            hotspot.predictive_variance
        ),
        fp_prompt_zyx=None,
        fp_prompt_model_xyz=None,
        sam_used=False,
        baseline_probability_zyx=(
            probability.copy()
        ),
        baseline_mask_zyx=(
            coarse.copy()
        ),
        final_mask_zyx=(
            coarse.copy()
        ),
        removal_mask_zyx=zeros.copy(),
        support_mask_zyx=zeros.copy(),
    )


def run_final_automatic_pipeline(
    baseline_model: torch.nn.Module,
    baseline_config: BaselineTrainingConfig,
    raw_mri_czyx: torch.Tensor,
    case_id: str,
    spacing_zyx_mm: Sequence[float],
    *,
    device: str | torch.device,
    t2f_path: str | Path | None = None,
    sam_model: torch.nn.Module | None = None,
) -> AutomaticPipelineResult:
    """Execute frozen NeuroPrompt3D_FinalAutomaticPolicy_v1."""

    raw_mri = _validate_raw_mri(
        raw_mri_czyx
    )

    spacing_zyx = _spacing3(
        spacing_zyx_mm
    )

    prepared_mri = (
        normalize_multimodal_foreground_zscore(
            raw_mri
        )
        .to(
            dtype=torch.float32
        )
        .contiguous()
    )

    if (
        prepared_mri.shape
        != raw_mri.shape
        or not torch.isfinite(
            prepared_mri
        ).all()
    ):
        raise AutomaticPipelineIntegrityError(
            "Baseline MRI preprocessing violated the frozen contract."
        )

    target_device = torch.device(
        device
    )

    baseline = run_deterministic_baseline(
        model=baseline_model,
        config=baseline_config,
        prepared_mri_czyx=prepared_mri,
        device=target_device,
    )

    spatial_shape = tuple(
        int(value)
        for value in raw_mri.shape[1:]
    )

    _validate_baseline_result(
        baseline,
        spatial_shape,
    )

    mc_batch = (
        prepared_mri
        .unsqueeze(0)
        .to(
            target_device
        )
    )

    mc = run_mc_dropout_case(
        model=baseline_model,
        mri_batch=mc_batch,
        roi_size=tuple(
            int(value)
            for value in baseline_config.patch_size
        ),
        overlap=float(
            baseline_config.validation_overlap
        ),
        case_id=case_id,
    )

    raw_numpy = (
        raw_mri
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    hotspot = select_uncertainty_hotspot(
        mri_czyx=raw_numpy,
        coarse_mask_zyx=(
            baseline.coarse_mask_zyx
        ),
        predictive_variance_zyx=(
            mc.predictive_variance
        ),
        spacing_zyx_mm=spacing_zyx,
    )

    if hotspot.abstention_condition is not None:
        return _baseline_result(
            baseline=baseline,
            mc=mc,
            hotspot=hotspot,
            gate_state=None,
            semantic_condition=(
                hotspot.abstention_condition
            ),
        )

    if (
        hotspot.coordinate_zyx is None
        or hotspot.predictive_variance is None
    ):
        raise AutomaticPipelineIntegrityError(
            "Valid hotspot result is missing required values."
        )

    gate_state = variance_gate(
        hotspot.predictive_variance
    )

    gate_action = final_action_for_gate(
        gate_state
    )

    if gate_state == GATE_ABSTAIN:
        if gate_action != ACTION_ABSTAIN_BASELINE:
            raise AutomaticPipelineIntegrityError(
                "ABSTAIN gate mapped to unexpected action."
            )

        return _baseline_result(
            baseline=baseline,
            mc=mc,
            hotspot=hotspot,
            gate_state=gate_state,
            semantic_condition=None,
        )

    if (
        gate_state != GATE_FP_ELIGIBLE
        or gate_action != ACTION_APPLY_LOCAL_FP
    ):
        raise AutomaticPipelineIntegrityError(
            "Variance gate returned an unexpected frozen state."
        )

    autoprompt = generate_autoprompt_v2(
        mri_czyx=raw_numpy,
        coarse_mask_zyx=(
            baseline.coarse_mask_zyx
        ),
        hotspot_zyx=(
            hotspot.coordinate_zyx
        ),
        spacing_zyx_mm=spacing_zyx,
    )

    fp_selection = (
        select_final_v1_fp_prompt(
            autoprompt
        )
    )

    if fp_selection.abstention_condition is not None:
        return _baseline_result(
            baseline=baseline,
            mc=mc,
            hotspot=hotspot,
            gate_state=gate_state,
            semantic_condition=(
                fp_selection
                .abstention_condition
            ),
        )

    fp_candidate = (
        fp_selection.candidate
    )

    if fp_candidate is None:
        raise AutomaticPipelineIntegrityError(
            "Final-v1 FP selection is malformed."
        )

    crop_points: dict[
        str,
        tuple[int, int, int],
    ] = {
        "uncertainty_hotspot": tuple(
            int(value)
            for value in hotspot.coordinate_zyx
        ),
        "false_positive_correction": tuple(
            int(value)
            for value in fp_candidate.coordinate_zyx
        ),
    }

    # Frozen AutoPrompt-v2 geometry may include the FN hypothesis.
    # It is crop geometry only and is never sent to SAM final-v1.
    if autoprompt.fn_candidate is not None:
        crop_points[
            "false_negative_correction"
        ] = tuple(
            int(value)
            for value in (
                autoprompt
                .fn_candidate
                .coordinate_zyx
            )
        )

    if t2f_path is None:
        raise AutomaticPipelineIntegrityError(
            "FP-eligible execution requires t2f_path."
        )

    try:
        prepared_sam = (
            prepare_sam_t2f_input(
                t2f_path=t2f_path,
                coarse_mask_zyx=(
                    baseline
                    .coarse_mask_zyx
                ),
                crop_prompt_points_zyx=(
                    crop_points
                ),
            )
        )

    except PromptPreservingCropInfeasible:
        return _baseline_result(
            baseline=baseline,
            mc=mc,
            hotspot=hotspot,
            gate_state=gate_state,
            semantic_condition=(
                SEMANTIC_CROP_INFEASIBLE
            ),
        )

    native_spacing_zyx = (
        _native_spacing_zyx_mm(
            prepared_sam
        )
    )

    if not np.allclose(
        native_spacing_zyx,
        spacing_zyx,
        rtol=0.0,
        atol=1e-6,
    ):
        raise AutomaticPipelineIntegrityError(
            "Frozen ZYX spacing does not match SAM input geometry."
        )

    if sam_model is None:
        raise AutomaticPipelineIntegrityError(
            "FP-eligible execution requires the frozen SAM model."
        )

    model_fp_xyz = (
        prepared_sam
        .model_points_xyz
        .get(
            "false_positive_correction"
        )
    )

    canonical_fp_xyz = (
        prepared_sam
        .canonical_points_xyz
        .get(
            "false_positive_correction"
        )
    )

    if (
        model_fp_xyz is None
        or canonical_fp_xyz is None
    ):
        raise AutomaticPipelineIntegrityError(
            "Prepared SAM input is missing the FP point."
        )

    sam_model = sam_model.to(
        target_device
    )

    if sam_model.training:
        raise AutomaticPipelineIntegrityError(
            "Frozen SAM model must be in eval mode."
        )

    if any(
        parameter.requires_grad
        for parameter in sam_model.parameters()
    ):
        raise AutomaticPipelineIntegrityError(
            "Frozen SAM model contains trainable parameters."
        )

    image_tensor = prepare_sam_image_tensor(
        prepared_sam.image_model_xyz,
        device=target_device,
    )

    image_embedding = encode_sam_image(
        sam_model,
        image_tensor,
    )

    sam_result = run_sam_point_branch(
        sam_model,
        image_embedding,
        coordinate_xyz=(
            model_fp_xyz
        ),
        label=SAM_NEGATIVE_LABEL,
    )

    if (
        sam_result.label
        != SAM_NEGATIVE_LABEL
        or tuple(
            int(value)
            for value in sam_result.coordinate_xyz
        )
        != tuple(
            int(value)
            for value in model_fp_xyz
        )
    ):
        raise AutomaticPipelineIntegrityError(
            "SAM result violates final-v1 FP prompt contract."
        )

    crop_policy = _crop_policy_mapping(
        prepared_sam.crop_metadata
    )

    full_geometry = (
        build_canonical_full_volume_geometry(
            baseline.coarse_mask_zyx,
            prepared_sam,
        )
    )

    fp_sam_full_xyz = (
        embed_model_mask_full(
            sam_result.mask_model_xyz,
            prepared_sam.canonical_shape_xyz,
            crop_policy,
        )
    )

    valid_fov_xyz = (
        fov_coverage_mask(
            prepared_sam.canonical_shape_xyz,
            crop_policy,
        )
    )

    (
        final_canonical_xyz,
        removal_canonical_xyz,
        support_canonical_xyz,
    ) = apply_local_fp_composite(
        coarse_mask=(
            full_geometry
            .coarse_mask_canonical_xyz
        ),
        fp_sam_mask=(
            fp_sam_full_xyz
        ),
        valid_sam_fov=(
            valid_fov_xyz
        ),
        fp_point_xyz=(
            canonical_fp_xyz
        ),
        spacing_xyz_mm=(
            full_geometry
            .spacing_xyz_mm
        ),
    )

    final_zyx = (
        canonical_mask_xyz_to_internal_zyx(
            final_canonical_xyz,
            prepared_sam,
        )
    )

    removal_zyx = (
        canonical_mask_xyz_to_internal_zyx(
            removal_canonical_xyz,
            prepared_sam,
        )
    )

    support_zyx = (
        canonical_mask_xyz_to_internal_zyx(
            support_canonical_xyz,
            prepared_sam,
        )
    )

    coarse_zyx = np.ascontiguousarray(
        baseline.coarse_mask_zyx,
        dtype=bool,
    )

    final_zyx = np.ascontiguousarray(
        final_zyx,
        dtype=bool,
    )

    removal_zyx = np.ascontiguousarray(
        removal_zyx,
        dtype=bool,
    )

    support_zyx = np.ascontiguousarray(
        support_zyx,
        dtype=bool,
    )

    if (
        final_zyx.shape != coarse_zyx.shape
        or removal_zyx.shape != coarse_zyx.shape
        or support_zyx.shape != coarse_zyx.shape
    ):
        raise AutomaticPipelineIntegrityError(
            "Final full-volume geometry is inconsistent."
        )

    if np.any(
        final_zyx
        & np.logical_not(
            coarse_zyx
        )
    ):
        raise AutomaticPipelineIntegrityError(
            "Final-v1 FP composite added foreground."
        )

    if np.any(
        removal_zyx
        & np.logical_not(
            support_zyx
        )
    ):
        raise AutomaticPipelineIntegrityError(
            "FP removal occurred outside frozen support."
        )

    expected_final = (
        coarse_zyx
        & np.logical_not(
            removal_zyx
        )
    )

    if not np.array_equal(
        final_zyx,
        expected_final,
    ):
        raise AutomaticPipelineIntegrityError(
            "Final mask is not exact removal-only baseline editing."
        )

    if not np.array_equal(
        final_zyx[
            np.logical_not(
                support_zyx
            )
        ],
        coarse_zyx[
            np.logical_not(
                support_zyx
            )
        ],
    ):
        raise AutomaticPipelineIntegrityError(
            "Final mask changed outside frozen support."
        )

    return AutomaticPipelineResult(
        action=ACTION_APPLY_LOCAL_FP,
        semantic_abstention_condition=None,
        gate_state=gate_state,
        case_seed=int(
            mc.case_seed
        ),
        mc_pass_hashes=tuple(
            mc.pass_hashes
        ),
        hotspot_zyx=tuple(
            int(value)
            for value in hotspot.coordinate_zyx
        ),
        hotspot_variance=float(
            hotspot.predictive_variance
        ),
        fp_prompt_zyx=tuple(
            int(value)
            for value in fp_candidate.coordinate_zyx
        ),
        fp_prompt_model_xyz=tuple(
            int(value)
            for value in model_fp_xyz
        ),
        sam_used=True,
        baseline_probability_zyx=(
            np.ascontiguousarray(
                baseline.probability_zyx,
                dtype=np.float32,
            )
        ),
        baseline_mask_zyx=(
            coarse_zyx.copy()
        ),
        final_mask_zyx=(
            final_zyx
        ),
        removal_mask_zyx=(
            removal_zyx
        ),
        support_mask_zyx=(
            support_zyx
        ),
    )
