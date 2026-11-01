"""Synthetic four-modality geometry tests; no BraTS voxel data."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.backend.services import frozen_slice_service as service


CASE_ID = "BraTS-GLI-02131-100"
MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def synthetic_case(tmp_path, monkeypatch, units="mm"):
    """Create four tiny placeholder paths and mock NIfTI headers."""
    case_dir = tmp_path / CASE_ID
    case_dir.mkdir()

    configuration = {}

    for modality in MODALITIES:
        path = case_dir / f"{CASE_ID}-{modality}.nii.gz"
        path.write_bytes(b"synthetic-test-placeholder")

        configuration[modality] = {
            "shape": (182, 218, 182),
            "affine": service.CANONICAL_AFFINE.copy(),
            "zooms": (1.0, 1.0, 1.0),
            "units": units,
        }

    def fake_load(filename):
        modality = Path(filename).name.split("-")[-1].split(".")[0]
        record = configuration[modality]

        header = SimpleNamespace(
            get_zooms=lambda: record["zooms"],
            get_xyzt_units=lambda: (record["units"], None),
        )

        return SimpleNamespace(
            shape=record["shape"],
            affine=record["affine"],
            header=header,
        )

    monkeypatch.setattr(service, "MRI_ROOT", tmp_path)
    monkeypatch.setattr(service.nib, "load", fake_load)

    return configuration, case_dir


def test_four_matching_mm_headers(tmp_path, monkeypatch):
    synthetic_case(tmp_path, monkeypatch, "mm")
    service._validate_case_mri_headers(CASE_ID)


def test_four_matching_unknown_headers(tmp_path, monkeypatch):
    synthetic_case(tmp_path, monkeypatch, "unknown")
    service._validate_case_mri_headers(CASE_ID)


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "mixed_units",
        "wrong_affine",
        "wrong_zooms",
        "wrong_shape",
        "nonfinite_affine",
        "unsupported_units",
    ],
)
def test_invalid_four_modality_geometry_fails_closed(
    tmp_path, monkeypatch, defect
):
    configuration, directory = synthetic_case(
        tmp_path, monkeypatch
    )

    record = configuration["t2f"]

    if defect == "missing":
        (directory / f"{CASE_ID}-t2f.nii.gz").unlink()
    elif defect == "mixed_units":
        record["units"] = "unknown"
    elif defect == "wrong_affine":
        record["affine"][0, 3] += 1.0
    elif defect == "wrong_zooms":
        record["zooms"] = (1.0, 1.0, 1.2)
    elif defect == "wrong_shape":
        record["shape"] = (181, 218, 182)
    elif defect == "nonfinite_affine":
        record["affine"][0, 0] = np.nan
    elif defect == "unsupported_units":
        record["units"] = "meter"

    with pytest.raises(service.FrozenSliceError):
        service._validate_case_mri_headers(CASE_ID)


def test_invalid_case_identity_rejected():
    with pytest.raises(service.FrozenSliceError):
        service._validate_case_mri_headers("../untrusted")
