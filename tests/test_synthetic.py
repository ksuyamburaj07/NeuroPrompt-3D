import nibabel as nib
import pytest
import torch

from src.data.nifti import load_case, save_case
from src.data.synthetic import create_synthetic_case
from src.visualization.slices import save_orthogonal_preview


def test_create_synthetic_case_returns_mri_and_mask():
    mri, mask = create_synthetic_case(shape=(24, 32, 40), seed=7)

    assert mri.shape == (1, 24, 32, 40)
    assert mask.shape == (24, 32, 40)
    assert mri.dtype == torch.float32
    assert mask.dtype == torch.uint8
    assert set(torch.unique(mask).tolist()) <= {0, 1}
    assert mask.sum() > 0


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
    mri = torch.zeros((1, 4, 5, 6), dtype=torch.float32)
    mask = torch.zeros((4, 5, 6), dtype=torch.uint8)
    mri[0, 1, 2, 3] = 0.75
    mask[1, 2, 3] = 1

    mri_path, mask_path = save_case(mri, mask, tmp_path, case_id="landmark")

    saved_mri = nib.load(mri_path)
    saved_mask = nib.load(mask_path)
    assert saved_mri.shape == (6, 5, 4)
    assert float(saved_mri.dataobj[3, 2, 1]) == 0.75
    assert int(saved_mask.dataobj[3, 2, 1]) == 1


def test_save_rejects_mismatched_spatial_shapes(tmp_path):
    mri = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
    mask = torch.zeros((8, 8, 7), dtype=torch.uint8)

    with pytest.raises(ValueError, match="spatial dimensions"):
        save_case(mri, mask, tmp_path, case_id="invalid")


def test_preview_is_saved(tmp_path):
    mri, mask = create_synthetic_case(shape=(24, 32, 40), seed=7)
    preview_path = tmp_path / "case_001_preview.png"

    result = save_orthogonal_preview(mri, mask, preview_path)

    assert result == preview_path
    assert preview_path.stat().st_size > 0
