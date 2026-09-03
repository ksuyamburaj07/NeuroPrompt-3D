"""Save and load NeuroPrompt-3D cases in NIfTI format."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.data.modalities import MODALITY_NAMES


def save_case(
    mri: torch.Tensor,
    mask: torch.Tensor,
    output_dir: str | Path,
    case_id: str,
) -> tuple[Path, Path]:
    """Save one MRI and mask pair and return their paths."""
    modality_count = len(MODALITY_NAMES)
    if mri.ndim != 4 or mri.shape[0] != modality_count:
        raise ValueError(
            f"MRI must have shape [{modality_count}, depth, height, width]"
        )
    if mask.ndim != 3:
        raise ValueError("Mask must have shape [depth, height, width]")
    if tuple(mri.shape[1:]) != tuple(mask.shape):
        raise ValueError("All MRI modalities and mask spatial dimensions must match")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mri_path = output_dir / f"{case_id}_mri.nii.gz"
    mask_path = output_dir / f"{case_id}_mask.nii.gz"
    affine = np.eye(4, dtype=np.float32)

    mri_array = mri.detach().cpu().permute(3, 2, 1, 0).contiguous().numpy()
    mask_array = mask.detach().cpu().permute(2, 1, 0).contiguous().numpy()
    nib.save(nib.Nifti1Image(mri_array, affine), mri_path)
    nib.save(nib.Nifti1Image(mask_array, affine), mask_path)

    return mri_path, mask_path


def load_case(
    mri_path: str | Path,
    mask_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one MRI and mask pair as PyTorch tensors."""
    mri_image = nib.load(mri_path)
    mask_image = nib.load(mask_path)
    mri_array = np.asarray(mri_image.dataobj, dtype=np.float32).copy()
    mask_array = np.asarray(mask_image.dataobj, dtype=np.uint8).copy()

    modality_count = len(MODALITY_NAMES)
    if mri_array.ndim != 4 or mri_array.shape[-1] != modality_count:
        raise ValueError(
            f"MRI NIfTI must have shape [x, y, z, {modality_count}]"
        )
    if mask_array.ndim != 3:
        raise ValueError("Mask NIfTI must have shape [x, y, z]")
    if tuple(mri_array.shape[:3]) != tuple(mask_array.shape):
        raise ValueError("All MRI modalities and mask spatial dimensions must match")
    if not np.allclose(mri_image.affine, mask_image.affine):
        raise ValueError("MRI and mask NIfTI affines must match")

    mri = torch.from_numpy(mri_array).permute(3, 2, 1, 0).contiguous()
    mask = torch.from_numpy(mask_array).permute(2, 1, 0).contiguous()
    return mri, mask
