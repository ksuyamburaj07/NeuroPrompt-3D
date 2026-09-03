"""Save and load NeuroPrompt-3D cases in NIfTI format."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def save_case(
    mri: torch.Tensor,
    mask: torch.Tensor,
    output_dir: str | Path,
    case_id: str,
) -> tuple[Path, Path]:
    """Save one MRI and mask pair and return their paths."""
    if mri.ndim != 4 or mri.shape[0] != 1:
        raise ValueError("MRI must have shape [1, depth, height, width]")
    if mask.ndim != 3:
        raise ValueError("Mask must have shape [depth, height, width]")
    if tuple(mri.shape[1:]) != tuple(mask.shape):
        raise ValueError("MRI and mask spatial dimensions must match")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mri_path = output_dir / f"{case_id}_mri.nii.gz"
    mask_path = output_dir / f"{case_id}_mask.nii.gz"
    affine = np.eye(4, dtype=np.float32)

    mri_array = (
        mri.squeeze(0).detach().cpu().permute(2, 1, 0).contiguous().numpy()
    )
    mask_array = mask.detach().cpu().permute(2, 1, 0).contiguous().numpy()
    nib.save(nib.Nifti1Image(mri_array, affine), mri_path)
    nib.save(nib.Nifti1Image(mask_array, affine), mask_path)

    return mri_path, mask_path


def load_case(
    mri_path: str | Path,
    mask_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one MRI and mask pair as PyTorch tensors."""
    mri_array = np.asarray(nib.load(mri_path).dataobj, dtype=np.float32).copy()
    mask_array = np.asarray(nib.load(mask_path).dataobj, dtype=np.uint8).copy()

    mri = torch.from_numpy(mri_array).permute(2, 1, 0).contiguous().unsqueeze(0)
    mask = torch.from_numpy(mask_array).permute(2, 1, 0).contiguous()
    return mri, mask
