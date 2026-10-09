"""Regression tests for aligned, read-only MRI segmentation overlays."""

from io import BytesIO

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.backend.core import paths
from app.backend.main import app
from app.backend.schemas.cases import (
    CaseValidationResponse,
    GeometryInfo,
    GroundTruthValidation,
    ModalityValidation,
)
from app.backend.services.run_service import (
    create_live_run,
    update_live_run,
)
from app.backend.services.viewer_overlay_service import (
    _aligned_mask,
)


def test_overlay_alignment_and_rejection(tmp_path, monkeypatch):
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"

    monkeypatch.setattr(
        paths, "LIVE_CASES_ROOT", cases_root
    )
    monkeypatch.setattr(
        paths, "LIVE_RUNS_ROOT", runs_root
    )
    _aligned_mask.cache_clear()

    case_id = "case_" + "a" * 32
    case_root = cases_root / case_id
    modality_root = case_root / "modalities"
    modality_root.mkdir(parents=True)

    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    mri = np.zeros((3, 4, 5), dtype=np.float32)
    mri[0, 1, 2] = 100.0

    for name in ("t1n", "t1c", "t2w", "t2f"):
        nib.save(
            nib.Nifti1Image(mri, affine),
            modality_root / f"{name}.nii.gz",
        )

    record = CaseValidationResponse(
        valid=True,
        status="ready",
        case_id=case_id,
        modalities={
            name: ModalityValidation(
                field=name,
                display_name=name.upper(),
                research_core_name=name,
                valid=True,
            )
            for name in ("t1n", "t1c", "t2w", "t2f")
        },
        geometry=GeometryInfo(
            shape_xyz=[3, 4, 5],
            tensor_shape_dhw=[5, 4, 3],
            voxel_spacing_xyz=[1.0, 1.0, 1.0],
            affine=affine.tolist(),
        ),
        ground_truth=GroundTruthValidation(
            provided=False
        ),
        ready_for_inference=True,
    )

    (case_root / "case.json").write_text(
        record.model_dump_json(),
        encoding="utf-8",
    )

    run = create_live_run(case_id)
    artifact_root = runs_root / run.run_id / "artifacts"
    artifact_root.mkdir(parents=True)

    baseline = np.zeros((3, 4, 5), dtype=np.uint8)
    baseline[0, 1, 2] = 1

    nib.save(
        nib.Nifti1Image(baseline, affine),
        artifact_root / "baseline_mask.nii.gz",
    )

    # Same shape but incorrect world-space position.
    incorrect_affine = affine.copy()
    incorrect_affine[0, 3] += 20.0

    nib.save(
        nib.Nifti1Image(baseline, incorrect_affine),
        artifact_root / "final_mask.nii.gz",
    )

    update_live_run(
        run.run_id,
        status="complete",
        stage="complete",
        progress=1.0,
        artifacts={
            "baseline_mask_nifti":
                "artifacts/baseline_mask.nii.gz",
            "final_mask_nifti":
                "artifacts/final_mask.nii.gz",
        },
    )

    base_url = (
        f"/api/v1/cases/{case_id}/viewer/"
        f"overlays/{run.run_id}"
    )

    with TestClient(app) as client:
        response = client.get(
            f"{base_url}/baseline/axial/2"
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["cache-control"] == "no-store"

        with Image.open(BytesIO(response.content)) as png:
            assert png.mode == "RGBA"
            assert png.size == (3, 4)

            # RAS reorientation flips original X.
            assert png.getpixel((2, 2))[3] == 255
            assert png.getpixel((0, 2))[3] == 0

        # The source mask must never be silently resampled.
        mismatch = client.get(
            f"{base_url}/final/axial/2"
        )
        assert mismatch.status_code == 409

        invalid_layer = client.get(
            f"{base_url}/unknown/axial/2"
        )
        assert invalid_layer.status_code == 400

        invalid_index = client.get(
            f"{base_url}/baseline/axial/5"
        )
        assert invalid_index.status_code == 400

        wrong_case = client.get(
            f"/api/v1/cases/case_{'b' * 32}/viewer/"
            f"overlays/{run.run_id}/baseline/axial/2"
        )
        assert wrong_case.status_code == 409

    _aligned_mask.cache_clear()
