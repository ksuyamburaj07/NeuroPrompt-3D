import nibabel as nib
import numpy as np
import pytest
import torch
import json

from src.data.dataset import (
    BraTSDataset,
    load_prepared_brats_case,
)

def test_load_prepared_brats_case_returns_model_ready_pair(tmp_path):
    case_id = "BraTS-GLI-00001-100"
    case_dir = tmp_path / case_id
    case_dir.mkdir()

    affine = np.eye(4)

    modality_suffixes = (
        "t1n",
        "t1c",
        "t2w",
        "t2f",
    )

    for index, suffix in enumerate(modality_suffixes):
        values = np.arange(
            1,
            25,
            dtype=np.float32,
        ).reshape(4, 3, 2)

        values += index * 100

        nib.save(
            nib.Nifti1Image(values, affine),
            case_dir / f"{case_id}-{suffix}.nii.gz",
        )

    segmentation = np.zeros(
        (4, 3, 2),
        dtype=np.uint8,
    )

    segmentation[1, 1, 1] = 1
    segmentation[2, 1, 1] = 2
    segmentation[3, 2, 1] = 3

    nib.save(
        nib.Nifti1Image(segmentation, affine),
        case_dir / f"{case_id}-seg.nii.gz",
    )

    mri, target = load_prepared_brats_case(
        case_dir,
        case_id,
    )

    assert mri.shape == (4, 2, 3, 4)
    assert mri.dtype == torch.float32

    assert target.shape == (2, 3, 4)
    assert target.dtype == torch.uint8

    assert sorted(target.unique().tolist()) == [0, 1]

    for channel in mri:
        assert torch.isclose(
            channel.mean(),
            torch.tensor(0.0),
            atol=1e-6,
        )

        assert torch.isclose(
            channel.std(correction=0),
            torch.tensor(1.0),
            atol=1e-6,
        )

def test_load_prepared_brats_case_rejects_segmentation_affine_mismatch(
    tmp_path,
):
    case_id = "BraTS-GLI-00001-100"
    case_dir = tmp_path / case_id
    case_dir.mkdir()

    mri_affine = np.eye(4)

    modality_suffixes = (
        "t1n",
        "t1c",
        "t2w",
        "t2f",
    )

    for suffix in modality_suffixes:
        values = np.ones(
            (4, 3, 2),
            dtype=np.float32,
        )

        nib.save(
            nib.Nifti1Image(
                values,
                mri_affine,
            ),
            case_dir / f"{case_id}-{suffix}.nii.gz",
        )

    segmentation = np.zeros(
        (4, 3, 2),
        dtype=np.uint8,
    )

    segmentation_affine = np.eye(4)
    segmentation_affine[0, 3] = 5.0

    nib.save(
        nib.Nifti1Image(
            segmentation,
            segmentation_affine,
        ),
        case_dir / f"{case_id}-seg.nii.gz",
    )

    with pytest.raises(
        ValueError,
        match="Segmentation NIfTI affine must match T1",
    ):
        load_prepared_brats_case(
            case_dir,
            case_id,
        )

def test_brats_dataset_reports_number_of_cases(tmp_path):
    case_ids = (
        "BraTS-GLI-00001-100",
        "BraTS-GLI-00002-100",
        "BraTS-GLI-00003-100",
    )

    dataset = BraTSDataset(
        cases_root=tmp_path,
        case_ids=case_ids,
    )

    assert len(dataset) == 3

def test_brats_dataset_getitem_loads_requested_case(tmp_path):
    case_id = "BraTS-GLI-00001-100"
    case_dir = tmp_path / case_id
    case_dir.mkdir()

    affine = np.eye(4)

    for index, suffix in enumerate(
        ("t1n", "t1c", "t2w", "t2f")
    ):
        values = np.arange(
            1,
            25,
            dtype=np.float32,
        ).reshape(4, 3, 2)

        values += index * 100

        nib.save(
            nib.Nifti1Image(values, affine),
            case_dir / f"{case_id}-{suffix}.nii.gz",
        )

    segmentation = np.zeros(
        (4, 3, 2),
        dtype=np.uint8,
    )

    segmentation[1, 1, 1] = 1
    segmentation[2, 1, 1] = 2

    nib.save(
        nib.Nifti1Image(segmentation, affine),
        case_dir / f"{case_id}-seg.nii.gz",
    )

    dataset = BraTSDataset(
        cases_root=tmp_path,
        case_ids=(case_id,),
    )

    mri, target = dataset[0]

    assert mri.shape == (4, 2, 3, 4)
    assert target.shape == (2, 3, 4)

    assert mri.dtype == torch.float32
    assert target.dtype == torch.uint8

    assert sorted(target.unique().tolist()) == [0, 1]

def test_brats_dataset_from_manifest_uses_frozen_case_ids(tmp_path):
    manifest_path = tmp_path / "splits.json"

    manifest = {
        "splits": {
            "train": {
                "case_ids": [
                    "BraTS-GLI-00008-101",
                    "BraTS-GLI-00005-100",
                ],
            },
            "validation": {
                "case_ids": [],
            },
            "test": {
                "case_ids": [],
            },
        },
    }

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    dataset = BraTSDataset.from_manifest(
        cases_root=tmp_path,
        manifest_path=manifest_path,
        split_name="train",
    )

    assert len(dataset) == 2

    assert dataset.case_ids == (
        "BraTS-GLI-00008-101",
        "BraTS-GLI-00005-100",
    )
