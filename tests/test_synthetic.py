from itertools import combinations

import nibabel as nib
import numpy as np
import pytest
import torch

from src.data.modalities import MODALITY_NAMES
from src.data.nifti import load_case, save_case
from src.data.synthetic import create_synthetic_case
from src.visualization.slices import save_multimodal_preview


def test_create_synthetic_case_returns_four_modalities_and_one_mask():
    mri, mask = create_synthetic_case(shape=(24, 32, 40), seed=7)

    assert MODALITY_NAMES == ("T1", "T1ce", "T2", "FLAIR")
    assert mri.shape == (4, 24, 32, 40)
    assert mask.shape == (24, 32, 40)
    assert mri.dtype == torch.float32
    assert mask.dtype == torch.uint8
    assert torch.isfinite(mri).all()
    assert mri.min() >= 0
    assert mri.max() <= 1
    assert set(torch.unique(mask).tolist()) <= {0, 1}
    assert mask.sum() > 0
    tumor_measurements = mri[:, mask.bool()]
    for first, second in combinations(range(len(MODALITY_NAMES)), 2):
        mean_difference = (tumor_measurements[first] - tumor_measurements[second]).abs().mean()
        assert mean_difference > 0.05

    assert tumor_measurements[1].mean() > tumor_measurements[0].mean()


def test_seed_makes_generation_repeatable():
    first_mri, first_mask = create_synthetic_case(shape=(16, 20, 24), seed=11)
    second_mri, second_mask = create_synthetic_case(shape=(16, 20, 24), seed=11)

    assert torch.equal(first_mri, second_mri)
    assert torch.equal(first_mask, second_mask)


def test_nifti_round_trip_preserves_case(tmp_path):
    mri, mask = create_synthetic_case(shape=(16, 20, 24), seed=11)

    mri_path, mask_path = save_case(mri, mask, tmp_path, case_id="case_001")
    loaded_mri, loaded_mask = load_case(mri_path, mask_path)

    assert mri_path.name == "case_001_mri.nii.gz"
    assert mask_path.name == "case_001_mask.nii.gz"
    assert torch.allclose(loaded_mri, mri)
    assert torch.equal(loaded_mask, mask)


def test_nifti_file_uses_xyz_axis_order(tmp_path):
    mri = torch.zeros((4, 4, 5, 6), dtype=torch.float32)
    mask = torch.zeros((4, 5, 6), dtype=torch.uint8)
    landmark_values = torch.tensor([0.25, 0.50, 0.75, 1.00])
    mri[:, 1, 2, 3] = landmark_values
    mask[1, 2, 3] = 1

    mri_path, mask_path = save_case(mri, mask, tmp_path, case_id="landmark")

    saved_mri = nib.load(mri_path)
    saved_mask = nib.load(mask_path)
    assert saved_mri.shape == (6, 5, 4, 4)
    for modality_index, expected in enumerate(landmark_values):
        assert float(saved_mri.dataobj[3, 2, 1, modality_index]) == expected
    assert int(saved_mask.dataobj[3, 2, 1]) == 1


def test_save_rejects_mismatched_spatial_shapes(tmp_path):
    mri = torch.zeros((4, 8, 8, 8), dtype=torch.float32)
    mask = torch.zeros((8, 8, 7), dtype=torch.uint8)

    with pytest.raises(ValueError, match="spatial dimensions"):
        save_case(mri, mask, tmp_path, case_id="invalid")


def test_save_rejects_wrong_modality_count(tmp_path):
    mri = torch.zeros((3, 8, 8, 8), dtype=torch.float32)
    mask = torch.zeros((8, 8, 8), dtype=torch.uint8)

    with pytest.raises(ValueError, match="MRI must have shape"):
        save_case(mri, mask, tmp_path, case_id="invalid")


def test_load_rejects_wrong_modality_count(tmp_path):
    affine = np.eye(4, dtype=np.float32)
    mri_path = tmp_path / "three_channels.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((6, 5, 4, 3)), affine), mri_path)
    nib.save(nib.Nifti1Image(np.zeros((6, 5, 4)), affine), mask_path)

    with pytest.raises(ValueError, match="MRI NIfTI must have shape"):
        load_case(mri_path, mask_path)


def test_load_rejects_mismatched_spatial_shapes(tmp_path):
    affine = np.eye(4, dtype=np.float32)
    mri_path = tmp_path / "mri.nii.gz"
    mask_path = tmp_path / "short_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((6, 5, 4, 4)), affine), mri_path)
    nib.save(nib.Nifti1Image(np.zeros((6, 5, 3)), affine), mask_path)

    with pytest.raises(ValueError, match="spatial dimensions"):
        load_case(mri_path, mask_path)


def test_load_rejects_mismatched_affines(tmp_path):
    mri_affine = np.eye(4, dtype=np.float32)
    mask_affine = np.eye(4, dtype=np.float32)
    mask_affine[0, 3] = 1
    mri_path = tmp_path / "mri.nii.gz"
    mask_path = tmp_path / "shifted_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((6, 5, 4, 4)), mri_affine), mri_path)
    nib.save(nib.Nifti1Image(np.zeros((6, 5, 4)), mask_affine), mask_path)

    with pytest.raises(ValueError, match="affines"):
        load_case(mri_path, mask_path)


def test_preview_is_saved(tmp_path):
    mri, mask = create_synthetic_case(shape=(24, 32, 40), seed=7)
    preview_path = tmp_path / "case_001_preview.png"

    result = save_multimodal_preview(mri, mask, preview_path)

    assert result == preview_path
    assert preview_path.stat().st_size > 0
