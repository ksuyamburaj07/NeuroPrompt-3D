"""Preprocessing utilities for NeuroPrompt-3D MRI volumes."""

import torch
from src.data.modalities import MODALITY_NAMES

def normalize_foreground_zscore(volume: torch.Tensor) -> torch.Tensor:
    """Z-score normalize nonzero voxels while preserving zero background."""
    if volume.ndim != 3:
        raise ValueError("Volume must be 3D [depth, height, width]")

    if not volume.is_floating_point():
        raise ValueError("Volume must use a floating-point dtype")

    foreground = volume != 0

    normalized = torch.zeros_like(volume)

    if not foreground.any():
        return normalized

    values = volume[foreground]
    mean = values.mean()
    std = values.std(correction=0)

    if std == 0:
        return normalized

    normalized[foreground] = (values - mean) / std

    return normalized

def normalize_multimodal_foreground_zscore(
    mri: torch.Tensor,
) -> torch.Tensor:
    """Normalize each MRI modality independently."""
    modality_count = len(MODALITY_NAMES)

    if mri.ndim != 4 or mri.shape[0] != modality_count:
        raise ValueError(
            f"MRI must have shape [{modality_count}, depth, height, width]"
        )

    normalized_channels = [
        normalize_foreground_zscore(channel)
        for channel in mri
    ]

    return torch.stack(normalized_channels, dim=0)

def whole_tumor_mask(
    segmentation: torch.Tensor,
) -> torch.Tensor:
    """Convert a multiclass segmentation into a binary whole-tumor mask."""
    if segmentation.ndim != 3:
        raise ValueError(
            "Segmentation must be 3D [depth, height, width]"
        )

    return (segmentation > 0).to(torch.uint8)

def prepare_model_case(
    mri: torch.Tensor,
    segmentation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare one MRI and segmentation pair for model training."""
    if tuple(mri.shape[1:]) != tuple(segmentation.shape):
        raise ValueError(
            "MRI and segmentation spatial dimensions must match"
        )

    normalized_mri = normalize_multimodal_foreground_zscore(
        mri
    )

    target = whole_tumor_mask(
        segmentation
    )

    return normalized_mri, target

def center_crop_pair(
    mri: torch.Tensor,
    target: torch.Tensor,
    spatial_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Center-crop MRI and target using the same spatial region."""
    crop_d, crop_h, crop_w = spatial_size

    depth, height, width = target.shape

    if (
       crop_d > depth
       or crop_h > height
       or crop_w > width
    ):
       raise ValueError(
           "Crop size must not exceed spatial dimensions"
       )

    start_d = (depth - crop_d) // 2
    start_h = (height - crop_h) // 2
    start_w = (width - crop_w) // 2

    end_d = start_d + crop_d
    end_h = start_h + crop_h
    end_w = start_w + crop_w

    cropped_mri = mri[
        :,
        start_d:end_d,
        start_h:end_h,
        start_w:end_w,
    ]

    cropped_target = target[
        start_d:end_d,
        start_h:end_h,
        start_w:end_w,
    ]

    return cropped_mri, cropped_target

def crop_pair_around_center(
    mri: torch.Tensor,
    target: torch.Tensor,
    center: tuple[int, int, int],
    spatial_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop MRI and target around the same spatial center."""
    center_d, center_h, center_w = center
    crop_d, crop_h, crop_w = spatial_size

    depth, height, width = target.shape

    if (
       crop_d > depth
       or crop_h > height
       or crop_w > width
    ):
       raise ValueError(
           "Crop size must not exceed spatial dimensions"
       )

    start_d = max(
        0,
    min(center_d - crop_d // 2, depth - crop_d),
    )

    start_h = max(
        0,
    min(center_h - crop_h // 2, height - crop_h),
    )

    start_w = max(
        0,
    min(center_w - crop_w // 2, width - crop_w),
    )

    end_d = start_d + crop_d
    end_h = start_h + crop_h
    end_w = start_w + crop_w

    cropped_mri = mri[
        :,
        start_d:end_d,
        start_h:end_h,
        start_w:end_w,
    ]

    cropped_target = target[
        start_d:end_d,
        start_h:end_h,
        start_w:end_w,
    ]

    return cropped_mri, cropped_target

def tumor_centered_crop(
    mri: torch.Tensor,
    target: torch.Tensor,
    spatial_size: tuple[int, int, int],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop MRI and target around a tumor voxel."""
    tumor_voxels = torch.nonzero(
        target > 0,
        as_tuple=False,
    )

    if tumor_voxels.numel() == 0:
        raise ValueError(
            "Target contains no tumor voxels"
        )

    index = torch.randint(
        len(tumor_voxels),
        size=(1,),
        generator=generator,
    ).item()

    center = tuple(
        int(value)
        for value in tumor_voxels[0]
    )

    return crop_pair_around_center(
        mri,
        target,
        center=center,
        spatial_size=spatial_size,
    )

def background_centered_crop(
    mri: torch.Tensor,
    target: torch.Tensor,
    spatial_size: tuple[int, int, int],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop around a non-tumor voxel inside MRI foreground."""
    brain_foreground = (mri != 0).any(dim=0)

    background_candidates = torch.nonzero(
        (target == 0) & brain_foreground,
        as_tuple=False,
    )

    if background_candidates.numel() == 0:
        raise ValueError(
             "No valid background voxels"
        )

    index = torch.randint(
        len(background_candidates),
        size=(1,),
        generator=generator,
    ).item()

    center = tuple(
        int(value)
        for value in background_candidates[0]
    )

    return crop_pair_around_center(
        mri,
        target,
        center=center,
        spatial_size=spatial_size,
    )


def sample_training_patch(
    mri: torch.Tensor,
    target: torch.Tensor,
    spatial_size: tuple[int, int, int],
    positive_probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample either a tumor-centered or background-centered training patch."""
    if not 0.0 <= positive_probability <= 1.0:
        raise ValueError(
            "positive_probability must be between 0 and 1"
        )

    choose_positive = (
        torch.rand(
            1,
            generator=generator,
        ).item()
        < positive_probability
    )

    if choose_positive:
        return tumor_centered_crop(
            mri,
            target,
            spatial_size=spatial_size,
            generator=generator,
        )

    return background_centered_crop(
        mri,
        target,
        spatial_size=spatial_size,
        generator=generator,
    )
