"""Visual checks for 3D MRI volumes and segmentation masks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_orthogonal_preview(
    mri: torch.Tensor,
    mask: torch.Tensor,
    output_path: str | Path,
) -> Path:
    """Save axial, coronal, and sagittal slices through the tumor center."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tumor_voxels = mask.nonzero(as_tuple=False)
    if len(tumor_voxels) > 0:
        center = tumor_voxels.float().mean(dim=0).round().to(torch.int64)
    else:
        center = torch.tensor([size // 2 for size in mask.shape])
    depth_index, height_index, width_index = center.tolist()

    mri_volume = mri.squeeze(0).detach().cpu().numpy()
    mask_volume = mask.detach().cpu().numpy()
    views = [
        ("Axial", mri_volume[depth_index], mask_volume[depth_index]),
        ("Coronal", mri_volume[:, height_index, :], mask_volume[:, height_index, :]),
        ("Sagittal", mri_volume[:, :, width_index], mask_volume[:, :, width_index]),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (title, image_slice, mask_slice) in zip(axes, views):
        axis.imshow(image_slice, cmap="gray", interpolation="nearest")
        overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
        axis.imshow(overlay, cmap="autumn", alpha=0.55, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle("Synthetic MRI with tumor-mask overlay (index space)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path
