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
