import torch
from torch import nn

from src.models.unet3d import LightweightUNet3D
from src.training.config import BaselineTrainingConfig
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader
from src.training.device import resolve_device

from src.data.loaders import (
    build_brats_training_dataloader,
    build_brats_validation_dataloader,
)

@dataclass
class BaselineExperiment:
    """Objects required for one baseline training experiment."""

    model: LightweightUNet3D
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader
    validation_loader: DataLoader
    config: BaselineTrainingConfig
    device: torch.device

def build_baseline_model(
    config: BaselineTrainingConfig,
) -> LightweightUNet3D:
    """Build the baseline 3D U-Net from the training config."""

    return LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config.base_channels,
        dropout_probability=config.dropout_probability,
    )


def build_baseline_optimizer(
    model: nn.Module,
    config: BaselineTrainingConfig,
) -> torch.optim.Optimizer:
    """Build the baseline Adam optimizer from the training config."""

    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

def build_baseline_dataloaders(
    cases_root: str | Path,
    manifest_path: str | Path,
    config: BaselineTrainingConfig,
):
    """Build frozen-split train and validation loaders from the config."""

    train_loader = build_brats_training_dataloader(
        cases_root=cases_root,
        manifest_path=manifest_path,
        spatial_size=config.patch_size,
        batch_size=config.batch_size,
        positive_probability=config.positive_probability,
        seed=config.seed,
    )

    validation_loader = build_brats_validation_dataloader(
        cases_root=cases_root,
        manifest_path=manifest_path,
    )

    return train_loader, validation_loader

def build_baseline_experiment(
    cases_root: str | Path,
    manifest_path: str | Path,
    config: BaselineTrainingConfig,
) -> BaselineExperiment:
    """Build the complete baseline training setup."""

    torch.manual_seed(config.seed)

    device = resolve_device(
        config.device,
    )

    model = build_baseline_model(
        config=config,
    )

    model = model.to(
        device=device,
    )

    optimizer = build_baseline_optimizer(
        model=model,
        config=config,
    )

    train_loader, validation_loader = build_baseline_dataloaders(
        cases_root=cases_root,
        manifest_path=manifest_path,
        config=config,
    )

    return BaselineExperiment(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        device=device,
    )
