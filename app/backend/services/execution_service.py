"""Execute one validated live case through the frozen NeuroPrompt-3D pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from app.backend.core import paths
from app.backend.core.device import (
    resolve_execution_device,
)
from app.backend.core.modalities import (
    APP_MODALITY_TO_CORE,
    REQUIRED_APP_MODALITIES,
)
from app.backend.schemas.runs import RunError
from app.backend.services.case_service import (
    load_live_case,
)
from app.backend.services.run_service import (
    load_live_run,
    update_live_run,
)
from src.data.nifti import load_multimodal_case
from src.inference.baseline import (
    load_baseline_checkpoint,
)
from src.pipeline.automatic import (
    run_final_automatic_pipeline,
)
from src.pipeline.policy import (
    FROZEN_VARIANCE_THRESHOLD,
)
from src.sam.loader import load_sam_checkpoint


def _derive_inference_case_id(
    raw_mri_czyx,
    affine: np.ndarray,
) -> str:
    """Derive stable live-case identity from canonical MRI content/geometry.

    The random application case UUID is deliberately not used for the
    frozen MC-Dropout seed. Identical canonical MRI data and geometry
    therefore produce the same frozen per-case seed across re-uploads.
    """

    raw = (
        raw_mri_czyx
        .detach()
        .to(device="cpu")
        .contiguous()
        .numpy()
    )

    if (
        raw.ndim != 4
        or int(raw.shape[0]) != 4
    ):
        raise ValueError(
            "Live inference identity requires MRI shape [4,D,H,W]."
        )

    if not np.isfinite(raw).all():
        raise ValueError(
            "Live inference identity requires finite MRI values."
        )

    affine_array = np.asarray(
        affine,
        dtype=np.float64,
    )

    if (
        affine_array.shape != (4, 4)
        or not np.isfinite(
            affine_array
        ).all()
    ):
        raise ValueError(
            "Live inference identity requires a finite 4x4 affine."
        )

    canonical_mri = np.ascontiguousarray(
        raw,
        dtype=np.dtype("<f4"),
    )

    canonical_affine = np.ascontiguousarray(
        affine_array,
        dtype=np.dtype("<f8"),
    )

    canonical_shape = np.asarray(
        canonical_mri.shape,
        dtype=np.dtype("<i8"),
    )

    digest = hashlib.sha256()

    digest.update(
        b"NeuroPrompt3D|M11|live-content-id|v1\0"
    )

    digest.update(
        memoryview(
            canonical_shape
        ).cast("B")
    )

    digest.update(
        memoryview(
            canonical_affine
        ).cast("B")
    )

    digest.update(
        memoryview(
            canonical_mri
        ).cast("B")
    )

    return (
        "live_sha256_"
        + digest.hexdigest()
    )


def _resolve_nifti(
    directory: Path,
    stem: str,
) -> Path:
    candidates = [
        directory / f"{stem}.nii.gz",
        directory / f"{stem}.nii",
    ]

    existing = [
        path
        for path in candidates
        if path.is_file()
    ]

    if len(existing) != 1:
        raise RuntimeError(
            f"Expected exactly one staged NIfTI for {stem!r}."
        )

    return existing[0]


def _case_modality_paths(
    case_id: str,
) -> dict[str, Path]:
    case = load_live_case(
        case_id
    )

    if not case.ready_for_inference:
        raise RuntimeError(
            "Live case is not ready for inference."
        )

    modality_root = (
        paths.LIVE_CASES_ROOT
        / case_id
        / "modalities"
    )

    return {
        field: _resolve_nifti(
            modality_root,
            field,
        )
        for field in REQUIRED_APP_MODALITIES
    }


def _save_internal_zyx_nifti(
    array_zyx: np.ndarray,
    destination: Path,
    *,
    affine: np.ndarray,
    dtype: np.dtype,
) -> None:
    array = np.asarray(
        array_zyx,
        dtype=dtype,
    )

    if array.ndim != 3:
        raise ValueError(
            "Artifact array must be 3D ZYX."
        )

    array_xyz = np.ascontiguousarray(
        array.transpose(2, 1, 0)
    )

    nib.save(
        nib.Nifti1Image(
            array_xyz,
            affine,
        ),
        destination,
    )


def _save_result_artifacts(
    *,
    run_id: str,
    case_id: str,
    inference_case_id: str,
    result,
    reference_path: Path,
    baseline_bundle,
    sam_bundle,
    execution_device: str,
) -> dict[str, str]:
    run_root = (
        paths.LIVE_RUNS_ROOT
        / run_id
    )

    artifact_root = (
        run_root
        / "artifacts"
    )

    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = nib.load(
        reference_path
    )

    affine = np.asarray(
        reference.affine,
        dtype=np.float64,
    )

    arrays = {
        "baseline_probability": (
            result.baseline_probability_zyx,
            np.float32,
        ),
        "baseline_mask": (
            result.baseline_mask_zyx,
            np.uint8,
        ),
        "final_mask": (
            result.final_mask_zyx,
            np.uint8,
        ),
        "removal_mask": (
            result.removal_mask_zyx,
            np.uint8,
        ),
        "support_mask": (
            result.support_mask_zyx,
            np.uint8,
        ),
    }

    artifacts: dict[str, str] = {}

    for name, (
        array,
        dtype,
    ) in arrays.items():
        npy_path = (
            artifact_root
            / f"{name}.npy"
        )

        np.save(
            npy_path,
            np.ascontiguousarray(
                array,
                dtype=dtype,
            ),
            allow_pickle=False,
        )

        artifacts[
            f"{name}_npy"
        ] = str(
            npy_path.relative_to(
                run_root
            )
        )

        nifti_path = (
            artifact_root
            / f"{name}.nii.gz"
        )

        _save_internal_zyx_nifti(
            array,
            nifti_path,
            affine=affine,
            dtype=np.dtype(dtype),
        )

        artifacts[
            f"{name}_nifti"
        ] = str(
            nifti_path.relative_to(
                run_root
            )
        )

    result_payload = {
        "run_id": run_id,
        "case_id": case_id,
        "inference_case_id": inference_case_id,
        "action": result.action,
        "gate_state": result.gate_state,
        "semantic_abstention_condition": (
            result.semantic_abstention_condition
        ),
        "case_seed": int(
            result.case_seed
        ),
        "mc_pass_hashes": list(
            result.mc_pass_hashes
        ),
        "hotspot_zyx": (
            list(result.hotspot_zyx)
            if result.hotspot_zyx is not None
            else None
        ),
        "hotspot_variance": (
            float(result.hotspot_variance)
            if result.hotspot_variance is not None
            else None
        ),
        "fp_prompt_zyx": (
            list(result.fp_prompt_zyx)
            if result.fp_prompt_zyx is not None
            else None
        ),
        "fp_prompt_model_xyz": (
            list(result.fp_prompt_model_xyz)
            if result.fp_prompt_model_xyz is not None
            else None
        ),
        "sam_used": bool(
            result.sam_used
        ),
        "frozen_variance_threshold": (
            FROZEN_VARIANCE_THRESHOLD
        ),
        "execution_device": execution_device,
        "baseline": {
            "checkpoint_sha256": (
                baseline_bundle
                .checkpoint_sha256
            ),
            "epoch": int(
                baseline_bundle.epoch
            ),
            "validation_loss": float(
                baseline_bundle
                .validation_loss
            ),
        },
        "sam": {
            "source_commit": (
                sam_bundle.source_commit
            ),
            "checkpoint_sha256": (
                sam_bundle
                .checkpoint_sha256
            ),
            "registry_key": (
                sam_bundle.registry_key
            ),
        },
        "predictive_variance_volume_saved": False,
    }

    result_path = (
        artifact_root
        / "result.json"
    )

    result_path.write_text(
        json.dumps(
            result_payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts["result_json"] = str(
        result_path.relative_to(
            run_root
        )
    )

    return artifacts


def execute_live_run(
    run_id: str,
) -> None:
    """Execute one run through the unchanged frozen final pipeline."""

    run = load_live_run(
        run_id
    )

    if run.status not in {
        "queued",
        "running",
    }:
        raise RuntimeError(
            "Only queued or running live runs can execute."
        )

    try:
        update_live_run(
            run_id,
            status="running",
            stage="validating",
            progress=0.05,
        )

        modality_paths = (
            _case_modality_paths(
                run.case_id
            )
        )

        core_paths = {
            APP_MODALITY_TO_CORE[field]:
                modality_paths[field]
            for field in REQUIRED_APP_MODALITIES
        }

        raw_mri = load_multimodal_case(
            core_paths
        )

        reference = nib.load(
            modality_paths["t1n"]
        )

        inference_case_id = (
            _derive_inference_case_id(
                raw_mri,
                reference.affine,
            )
        )

        spacing_xyz = tuple(
            float(value)
            for value
            in reference.header.get_zooms()[:3]
        )

        spacing_zyx = tuple(
            reversed(
                spacing_xyz
            )
        )

        device = (
            resolve_execution_device()
        )

        update_live_run(
            run_id,
            status="running",
            stage="loading_models",
            progress=0.15,
            execution_device=str(
                device
            ),
            inference_case_id=(
                inference_case_id
            ),
        )

        baseline_bundle = (
            load_baseline_checkpoint(
                paths.BASELINE_CHECKPOINT
            )
        )

        # Load the frozen SAM model up front.
        #
        # This preserves the single authoritative frozen
        # run_final_automatic_pipeline execution. We do not
        # duplicate baseline/MC/hotspot/policy execution merely
        # to discover whether SAM will later be needed.
        sam_bundle = (
            load_sam_checkpoint(
                paths.SAM_SOURCE_ROOT,
                paths.SAM_CHECKPOINT,
            )
        )

        update_live_run(
            run_id,
            status="running",
            stage="automatic_pipeline",
            progress=0.25,
            execution_device=str(
                device
            ),
        )

        result = (
            run_final_automatic_pipeline(
                baseline_model=(
                    baseline_bundle.model
                ),
                baseline_config=(
                    baseline_bundle.config
                ),
                raw_mri_czyx=raw_mri,
                case_id=inference_case_id,
                spacing_zyx_mm=(
                    spacing_zyx
                ),
                device=device,
                t2f_path=(
                    modality_paths["t2f"]
                ),
                sam_model=(
                    sam_bundle.model
                ),
            )
        )

        update_live_run(
            run_id,
            status="running",
            stage="finalizing",
            progress=0.90,
            execution_device=str(
                device
            ),
        )

        artifacts = (
            _save_result_artifacts(
                run_id=run_id,
                case_id=run.case_id,
                inference_case_id=(
                    inference_case_id
                ),
                result=result,
                reference_path=(
                    modality_paths["t1n"]
                ),
                baseline_bundle=(
                    baseline_bundle
                ),
                sam_bundle=sam_bundle,
                execution_device=str(
                    device
                ),
            )
        )

        update_live_run(
            run_id,
            status="complete",
            stage="complete",
            progress=1.0,
            execution_device=str(
                device
            ),
            inference_case_id=(
                inference_case_id
            ),
            mc_case_seed=int(
                result.case_seed
            ),
            action=result.action,
            gate_state=result.gate_state,
            hotspot_zyx=(
                list(
                    result.hotspot_zyx
                )
                if result.hotspot_zyx
                is not None
                else None
            ),
            hotspot_variance=(
                float(
                    result.hotspot_variance
                )
                if result.hotspot_variance
                is not None
                else None
            ),
            fp_prompt_zyx=(
                list(
                    result.fp_prompt_zyx
                )
                if result.fp_prompt_zyx
                is not None
                else None
            ),
            fp_prompt_model_xyz=(
                list(
                    result.fp_prompt_model_xyz
                )
                if result.fp_prompt_model_xyz
                is not None
                else None
            ),
            sam_used=bool(
                result.sam_used
            ),
            sam_refinement_skipped=(
                not bool(
                    result.sam_used
                )
            ),
            semantic_abstention_condition=(
                result
                .semantic_abstention_condition
            ),
            artifacts=artifacts,
        )

    except Exception as exc:
        update_live_run(
            run_id,
            status="failed",
            stage="failed",
            error=RunError(
                code=(
                    exc.__class__.__name__
                ),
                message=str(exc),
            ),
        )

        raise
