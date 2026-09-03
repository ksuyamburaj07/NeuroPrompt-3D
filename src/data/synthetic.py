"""Create small synthetic 3D MRI cases for development and tests."""

import torch

from src.data.modalities import MODALITY_NAMES


def create_synthetic_case(
    shape: tuple[int, int, int] = (64, 64, 64),
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return four aligned MRI modalities and one binary tumor mask."""
    depth, height, width = shape
    generator = torch.Generator().manual_seed(seed)

    z, y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, depth),
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )

    brain = (x / 0.82) ** 2 + (y / 0.90) ** 2 + (z / 0.78) ** 2 <= 1
    tumor_distance = (
        ((x - 0.22) / 0.18) ** 2
        + ((y + 0.10) / 0.14) ** 2
        + ((z - 0.05) / 0.16) ** 2
    )
    tumor = brain & (tumor_distance <= 1)
    enhancing_rim = tumor & (tumor_distance >= 0.50)
    tumor_core = tumor & ~enhancing_rim
    mask = tumor.to(torch.uint8)

    radial_distance = (x**2 + y**2 + z**2).clamp(max=1)
    anatomy = 1 - radial_distance
    tumor_signal = tumor.to(torch.float32)
    rim_signal = enhancing_rim.to(torch.float32)
    core_signal = tumor_core.to(torch.float32)

    modality_signals = torch.stack(
        (
            0.28 + 0.38 * anatomy - 0.10 * tumor_signal,
            0.25 + 0.34 * anatomy + 0.45 * rim_signal + 0.12 * core_signal,
            0.18 + 0.28 * anatomy + 0.42 * tumor_signal,
            0.15 + 0.25 * anatomy + 0.55 * tumor_signal,
        ),
        dim=0,
    )
    noise = torch.randn((len(MODALITY_NAMES), *shape), generator=generator) * 0.03
    mri = torch.where(
        brain.unsqueeze(0),
        modality_signals + noise,
        torch.zeros_like(modality_signals),
    ).clamp(0, 1)

    return mri.to(torch.float32), mask
