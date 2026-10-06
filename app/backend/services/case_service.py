"""Live-case staging and validation for the M11 application layer."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import nibabel as nib
import torch
from fastapi import UploadFile

from app.backend.core import paths
from app.backend.core.modalities import (
    APP_MODALITY_LABELS,
    APP_MODALITY_TO_CORE,
    REQUIRED_APP_MODALITIES,
)
from app.backend.schemas.cases import (
    CaseValidationResponse,
    GeometryInfo,
    GroundTruthValidation,
    ModalityValidation,
    ValidationIssue,
)
from src.data.nifti import load_multimodal_case, load_segmentation


_CASE_ID_PATTERN = re.compile(r"^case_[0-9a-f]{32}$")
_COPY_CHUNK_SIZE = 8 * 1024 * 1024


def _nifti_suffix(filename: str | None) -> str | None:
    if not filename:
        return None

    lowered = filename.lower()

    if lowered.endswith(".nii.gz"):
        return ".nii.gz"

    if lowered.endswith(".nii"):
        return ".nii"

    return None


def _new_modality_statuses(
    uploads: Mapping[str, UploadFile | None],
) -> dict[str, ModalityValidation]:
    return {
        field: ModalityValidation(
            field=field,
            display_name=APP_MODALITY_LABELS[field],
            research_core_name=APP_MODALITY_TO_CORE[field],
            filename=(
                uploads[field].filename
                if uploads.get(field) is not None
                else None
            ),
            valid=None,
        )
        for field in REQUIRED_APP_MODALITIES
    }


def _invalid_response(
    *,
    modalities: dict[str, ModalityValidation],
    ground_truth: GroundTruthValidation,
    errors: list[ValidationIssue],
) -> CaseValidationResponse:
    return CaseValidationResponse(
        valid=False,
        status="invalid",
        case_id=None,
        modalities=modalities,
        geometry=None,
        ground_truth=ground_truth,
        ready_for_inference=False,
        errors=errors,
    )


async def _copy_upload(
    upload: UploadFile,
    destination: Path,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0

    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(_COPY_CHUNK_SIZE)

            if not chunk:
                break

            output.write(chunk)
            total_bytes += len(chunk)

    await upload.seek(0)

    return total_bytes


def _field_from_core_error(message: str) -> str | None:
    # Check longer core names first so T1 does not accidentally match T1ce.
    pairs = sorted(
        APP_MODALITY_TO_CORE.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    for app_field, core_name in pairs:
        if f"{core_name} NIfTI" in message:
            return app_field

    if "Segmentation" in message:
        return "segmentation"

    return None


def _validate_case_id(case_id: str) -> None:
    if not _CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError("Invalid live case identifier")


async def validate_and_stage_case(
    *,
    uploads: Mapping[str, UploadFile | None],
    segmentation: UploadFile | None,
) -> CaseValidationResponse:
    """Validate one user-uploaded case and persist it only if valid."""

    modalities = _new_modality_statuses(uploads)

    gt_status = GroundTruthValidation(
        provided=segmentation is not None,
        filename=segmentation.filename if segmentation is not None else None,
        valid=None,
    )

    issues: list[ValidationIssue] = []

    # --------------------------------------------------------------
    # Presence + extension validation
    # --------------------------------------------------------------
    for field in REQUIRED_APP_MODALITIES:
        upload = uploads.get(field)

        if upload is None or not upload.filename:
            modalities[field].valid = False
            issues.append(
                ValidationIssue(
                    code="missing_modality",
                    field=field,
                    message=(
                        f"{APP_MODALITY_LABELS[field]} is required."
                    ),
                )
            )
            continue

        if _nifti_suffix(upload.filename) is None:
            modalities[field].valid = False
            issues.append(
                ValidationIssue(
                    code="invalid_file_extension",
                    field=field,
                    message=(
                        f"{APP_MODALITY_LABELS[field]} must be "
                        "a .nii or .nii.gz file."
                    ),
                )
            )

    if segmentation is not None:
        if _nifti_suffix(segmentation.filename) is None:
            gt_status.valid = False
            issues.append(
                ValidationIssue(
                    code="invalid_file_extension",
                    field="segmentation",
                    message=(
                        "Ground-truth segmentation must be "
                        "a .nii or .nii.gz file."
                    ),
                )
            )

    if issues:
        return _invalid_response(
            modalities=modalities,
            ground_truth=gt_status,
            errors=issues,
        )

    # --------------------------------------------------------------
    # Temporary staging
    # --------------------------------------------------------------
    paths.ensure_runtime_directories()

    case_id = f"case_{uuid4().hex}"

    staging_root = (
        paths.LIVE_CASES_ROOT
        / f".staging-{case_id}"
    )

    final_root = paths.LIVE_CASES_ROOT / case_id

    staging_root.mkdir(parents=True, exist_ok=False)

    staged_modalities: dict[str, Path] = {}

    try:
        # ----------------------------------------------------------
        # Copy uploads using fixed application-owned filenames.
        # Original filenames are metadata only and never become paths.
        # ----------------------------------------------------------
        for field in REQUIRED_APP_MODALITIES:
            upload = uploads[field]

            assert upload is not None

            suffix = _nifti_suffix(upload.filename)

            assert suffix is not None

            destination = (
                staging_root
                / "modalities"
                / f"{field}{suffix}"
            )

            size = await _copy_upload(
                upload,
                destination,
            )

            if size == 0:
                raise ValueError(
                    f"{APP_MODALITY_LABELS[field]} upload is empty"
                )

            staged_modalities[field] = destination

        staged_segmentation: Path | None = None

        if segmentation is not None:
            suffix = _nifti_suffix(segmentation.filename)

            assert suffix is not None

            staged_segmentation = (
                staging_root
                / f"segmentation{suffix}"
            )

            size = await _copy_upload(
                segmentation,
                staged_segmentation,
            )

            if size == 0:
                raise ValueError(
                    "Ground-truth segmentation upload is empty"
                )

        # ----------------------------------------------------------
        # Canonical frozen-core MRI validation.
        # ----------------------------------------------------------
        core_paths = {
            APP_MODALITY_TO_CORE[field]:
                staged_modalities[field]
            for field in REQUIRED_APP_MODALITIES
        }

        try:
            mri = load_multimodal_case(core_paths)
        except Exception as exc:
            message = str(exc)
            field = _field_from_core_error(message)

            if field in modalities:
                modalities[field].valid = False

            issues.append(
                ValidationIssue(
                    code="nifti_validation_failed",
                    field=field,
                    message=message,
                )
            )

            shutil.rmtree(
                staging_root,
                ignore_errors=True,
            )

            return _invalid_response(
                modalities=modalities,
                ground_truth=gt_status,
                errors=issues,
            )

        for status in modalities.values():
            status.valid = True

        # ----------------------------------------------------------
        # Report geometry from the canonical reference modality T1n.
        # ----------------------------------------------------------
        reference_image = nib.load(
            staged_modalities["t1n"]
        )

        geometry = GeometryInfo(
            shape_xyz=[
                int(value)
                for value in reference_image.shape
            ],
            tensor_shape_dhw=[
                int(value)
                for value in mri.shape[1:]
            ],
            voxel_spacing_xyz=[
                float(value)
                for value in reference_image.header.get_zooms()[:3]
            ],
            affine=[
                [float(value) for value in row]
                for row in reference_image.affine.tolist()
            ],
        )

        # ----------------------------------------------------------
        # Optional GT validation.
        # Reuse canonical GT loader, then explicitly enforce the same
        # spatial compatibility required by model preprocessing.
        # ----------------------------------------------------------
        if staged_segmentation is not None:
            try:
                target = load_segmentation(
                    staged_segmentation,
                    reference_path=staged_modalities["t1n"],
                )

                if tuple(target.shape) != tuple(mri.shape[1:]):
                    raise ValueError(
                        "Segmentation spatial dimensions must match MRI"
                    )

            except Exception as exc:
                gt_status.valid = False

                issues.append(
                    ValidationIssue(
                        code="ground_truth_validation_failed",
                        field="segmentation",
                        message=str(exc),
                    )
                )

                shutil.rmtree(
                    staging_root,
                    ignore_errors=True,
                )

                return CaseValidationResponse(
                    valid=False,
                    status="invalid",
                    case_id=None,
                    modalities=modalities,
                    geometry=geometry,
                    ground_truth=gt_status,
                    ready_for_inference=False,
                    errors=issues,
                )

            gt_status.valid = True
            gt_status.tensor_shape_dhw = [
                int(value)
                for value in target.shape
            ]
            gt_status.labels = [
                int(value)
                for value in torch.unique(target).tolist()
            ]

        # ----------------------------------------------------------
        # Persist metadata only after successful validation.
        # ----------------------------------------------------------
        result = CaseValidationResponse(
            valid=True,
            status="ready",
            case_id=case_id,
            modalities=modalities,
            geometry=geometry,
            ground_truth=gt_status,
            ready_for_inference=True,
            errors=[],
        )

        metadata_path = staging_root / "case.json"

        metadata_path.write_text(
            json.dumps(
                result.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        staging_root.rename(final_root)

        return result

    except Exception:
        shutil.rmtree(
            staging_root,
            ignore_errors=True,
        )
        raise


def load_live_case(
    case_id: str,
) -> CaseValidationResponse:
    _validate_case_id(case_id)

    metadata_path = (
        paths.LIVE_CASES_ROOT
        / case_id
        / "case.json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(case_id)

    return CaseValidationResponse.model_validate_json(
        metadata_path.read_text(encoding="utf-8")
    )


def delete_live_case(
    case_id: str,
) -> bool:
    _validate_case_id(case_id)

    case_root = paths.LIVE_CASES_ROOT / case_id

    if not case_root.is_dir():
        raise FileNotFoundError(case_id)

    shutil.rmtree(case_root)

    return True
