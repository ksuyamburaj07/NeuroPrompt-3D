"""Visual checks for 3D MRI volumes and segmentation masks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.modalities import MODALITY_NAMES


def save_multimodal_preview(
    mri: torch.Tensor,
    mask: torch.Tensor,
    output_path: str | Path,
) -> Path:
    """Save all MRI modalities at an axial slice through the tumor center."""
    if mri.ndim != 4 or mri.shape[0] != len(MODALITY_NAMES):
        raise ValueError("MRI must contain T1, T1ce, T2, and FLAIR channels")
    if mask.ndim != 3 or tuple(mri.shape[1:]) != tuple(mask.shape):
        raise ValueError("MRI and mask spatial dimensions must match")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tumor_voxels = mask.nonzero(as_tuple=False)
    if len(tumor_voxels) > 0:
        center = tumor_voxels.float().mean(dim=0).round().to(torch.int64)
    else:
        center = torch.tensor([size // 2 for size in mask.shape])
    depth_index = int(center[0])

    mri_volume = mri.detach().cpu().numpy()
    mask_slice = mask.detach().cpu().numpy()[depth_index]
    overlay = np.ma.masked_where(mask_slice == 0, mask_slice)

    figure, axes = plt.subplots(2, 2, figsize=(9, 9))
    for modality_index, (axis, modality_name) in enumerate(
        zip(axes.flat, MODALITY_NAMES)
    ):
        axis.imshow(
            mri_volume[modality_index, depth_index],
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.imshow(overlay, cmap="autumn", alpha=0.55, interpolation="nearest")
        axis.set_title(modality_name)
        axis.axis("off")

    figure.suptitle(
        f"Aligned synthetic MRI modalities with tumor overlay · axial slice {depth_index}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path
