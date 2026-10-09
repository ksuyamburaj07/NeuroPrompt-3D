"""Tests for read-only, RAS-oriented MRI slice serving."""

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
from app.backend.services.viewer_service import _load_ras


def test_viewer_preserves_anatomical_orientation(
    tmp_path,
    monkeypatch,
):
    case_id = "case_" + "a" * 32
    root = tmp_path / "live_cases"
    case_root = root / case_id
    modalities = case_root / "modalities"
    modalities.mkdir(parents=True)

    monkeypatch.setattr(paths, "LIVE_CASES_ROOT", root)
    _load_ras.cache_clear()

    original = np.zeros((3, 4, 5), dtype=np.float32)
    original[0, 1, 2] = 1000
    original[2, 1, 2] = 100

    affine = np.diag([-1.0, 1.0, 1.0, 1.0])

    for modality in ("t1n", "t1c", "t2w", "t2f"):
        nib.save(
            nib.Nifti1Image(original, affine),
            modalities / f"{modality}.nii.gz",
        )

    record = CaseValidationResponse(
        valid=True,
        status="ready",
        case_id=case_id,
        modalities={
            key: ModalityValidation(
                field=key,
                display_name=key.upper(),
                research_core_name=key,
                valid=True,
            )
            for key in ("t1n", "t1c", "t2w", "t2f")
        },
        geometry=GeometryInfo(
            shape_xyz=[3, 4, 5],
            tensor_shape_dhw=[5, 4, 3],
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

    with TestClient(app) as client:
        base = f"/api/v1/cases/{case_id}/viewer"

        metadata = client.get(f"{base}/metadata")
        assert metadata.status_code == 200

        info = metadata.json()
        assert info["orientation_convention"] == "RAS+"
        assert info["original_axis_codes"] == ["L", "A", "S"]
        assert info["shape_ras_xyz"] == [3, 4, 5]

        response = client.get(
            f"{base}/slices/t2f/axial/2"
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["cache-control"] == "no-store"

        with Image.open(BytesIO(response.content)) as image:
            assert image.size == (3, 4)

            # Original X axis points left in world space.
            # RAS reorientation must reverse its display order.
            assert image.getpixel((2, 2)) > image.getpixel((0, 2))

        out_of_range = client.get(
            f"{base}/slices/t2f/axial/5"
        )
        assert out_of_range.status_code == 400

        missing = client.get(
            f"/api/v1/cases/case_{'b' * 32}/viewer/metadata"
        )
        assert missing.status_code == 404

    _load_ras.cache_clear()
