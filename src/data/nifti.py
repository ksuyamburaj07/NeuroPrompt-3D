"""Save and load NeuroPrompt-3D cases in NIfTI format."""

from collections.abc import Mapping
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.data.cases import ordered_modality_paths
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


def load_multimodal_case(
    paths_by_modality: Mapping[str, str | Path | None],
) -> torch.Tensor:
    """Load separate 3D NIfTI files as contiguous float32 [4, D, H, W].

    Channels follow MODALITY_NAMES. Each file stores [X, Y, Z], so the
    returned spatial axes are [Z, Y, X], matching load_case. Shapes must
    match exactly; affines must match T1 with atol=1e-5 and rtol=0.
    """
    paths = ordered_modality_paths(paths_by_modality)
    images = [nib.load(path) for path in paths]
    reference = images[0]

    for modality, image in zip(MODALITY_NAMES, images):
        if len(image.shape) != 3:
            raise ValueError(
                f"{modality} NIfTI must be 3D [X, Y, Z]; got shape {image.shape}"
            )
        if image.shape != reference.shape:
            raise ValueError(
                f"{modality} NIfTI spatial shape {image.shape} must match "
                f"{MODALITY_NAMES[0]} {reference.shape}"
            )
        if not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5):
            raise ValueError(
                f"{modality} NIfTI affine must match {MODALITY_NAMES[0]} "
                "(atol=1e-5, rtol=0)"
            )

    arrays = [np.asarray(image.dataobj, dtype=np.float32) for image in images]
    mri_array = np.stack(arrays, axis=0)
    return torch.from_numpy(mri_array).permute(0, 3, 2, 1).contiguous()

def load_segmentation(
    path: str | Path,
    reference_path: str | Path | None = None,
) -> torch.Tensor:
    """Load one 3D segmentation NIfTI as contiguous uint8 [D, H, W]."""
    image = nib.load(path)

    if len(image.shape) != 3:
        raise ValueError(
            f"Segmentation NIfTI must be 3D [X, Y, Z]; got shape {image.shape}"
        )

    if reference_path is not None:
        reference = nib.load(reference_path)

        if not np.allclose(
            image.affine,
            reference.affine,
            rtol=0,
            atol=1e-5,
        ):
            raise ValueError(
                "Segmentation NIfTI affine must match T1 "
                "(atol=1e-5, rtol=0)"
            )

    array = np.asarray(
        image.dataobj,
        dtype=np.uint8,
    )

    return (
        torch.from_numpy(array)
        .permute(2, 1, 0)
        .contiguous()
    )
