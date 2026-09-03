"""Create small synthetic 3D MRI cases for development and tests."""

import torch


def create_synthetic_case(
    shape: tuple[int, int, int] = (64, 64, 64),
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an MRI tensor and its binary tumor mask."""
    depth, height, width = shape
    generator = torch.Generator().manual_seed(seed)

    z, y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, depth),
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )

    brain = (x / 0.82) ** 2 + (y / 0.90) ** 2 + (z / 0.78) ** 2 <= 1
    tumor = (
        ((x - 0.22) / 0.18) ** 2
        + ((y + 0.10) / 0.14) ** 2
        + ((z - 0.05) / 0.16) ** 2
        <= 1
    )
    mask = (brain & tumor).to(torch.uint8)

    noise = torch.randn(shape, generator=generator) * 0.04
    anatomy = 0.35 + 0.25 * (1 - (x**2 + y**2 + z**2).clamp(max=1))
    mri = torch.where(brain, anatomy + noise, torch.zeros_like(noise))
    mri = (mri + mask.float() * 0.35).clamp(0, 1).unsqueeze(0)

    return mri.to(torch.float32), mask
