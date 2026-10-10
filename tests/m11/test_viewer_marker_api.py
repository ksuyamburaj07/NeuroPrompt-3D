"""Tests for frozen hotspot and prompt coordinate visualization."""

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

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
from app.backend.services.viewer_marker_service import marker_to_ras


def test_native_zyx_to_ras_handles_flip_and_permutation():
    array = np.zeros((5, 6, 7), dtype=np.float32)

    left_anterior_superior = nib.Nifti1Image(
        array,
        np.diag([-1.0, 1.0, 1.0, 1.0]),
    )

    marker = marker_to_ras([2, 3, 0], left_anterior_superior)
    assert marker["ras_xyz"] == [4, 3, 2]

    # Native X points posterior, native Y points right.
    permuted = nib.Nifti1Image(
        array,
        np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    )

    marker = marker_to_ras([2, 3, 0], permuted)
    assert marker["ras_xyz"] == [3, 4, 2]


def test_marker_api_uses_completed_matching_case(tmp_path, monkeypatch):
    case_root_base = tmp_path / "cases"
    run_root_base = tmp_path / "runs"

    monkeypatch.setattr(
        paths, "LIVE_CASES_ROOT", case_root_base
    )
    monkeypatch.setattr(
        paths, "LIVE_RUNS_ROOT", run_root_base
    )

    case_id = "case_" + "a" * 32
    case_root = case_root_base / case_id
    modality_root = case_root / "modalities"
    modality_root.mkdir(parents=True)

    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    values = np.zeros((5, 6, 7), dtype=np.float32)

    nib.save(
        nib.Nifti1Image(values, affine),
        modality_root / "t1n.nii.gz",
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
            shape_xyz=[5, 6, 7],
            tensor_shape_dhw=[7, 6, 5],
            voxel_spacing_xyz=[1.0, 1.0, 1.0],
            affine=affine.tolist(),
        ),
        ground_truth=GroundTruthValidation(provided=False),
        ready_for_inference=True,
    )

    (case_root / "case.json").write_text(
        record.model_dump_json(),
        encoding="utf-8",
    )

    run = create_live_run(case_id)

    url = (
        f"/api/v1/cases/{case_id}/viewer/"
        f"markers/{run.run_id}"
    )

    with TestClient(app) as client:
        pending = client.get(url)
        assert pending.status_code == 409

        update_live_run(
            run.run_id,
            status="complete",
            stage="complete",
            progress=1.0,
            hotspot_zyx=[2, 3, 0],
            hotspot_variance=0.25,
            fp_prompt_zyx=[1, 2, 2],
        )

        response = client.get(url)
        assert response.status_code == 200

        data = response.json()
        assert data["hotspot"]["source_zyx"] == [2, 3, 0]
        assert data["hotspot"]["ras_xyz"] == [4, 3, 2]
        assert data["negative_prompt"]["ras_xyz"] == [2, 2, 1]
        assert data["hotspot_variance"] == 0.25

        wrong_case = client.get(
            f"/api/v1/cases/case_{'b' * 32}/viewer/"
            f"markers/{run.run_id}"
        )
        assert wrong_case.status_code == 409

        update_live_run(
            run.run_id,
            hotspot_zyx=[99, 3, 0],
        )
        bad_coordinate = client.get(url)
        assert bad_coordinate.status_code == 409
