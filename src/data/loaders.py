from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import (
    BraTSDataset,
    TrainingPatchDataset,
)


def build_training_dataloader(
    base_dataset: Dataset,
    spatial_size: tuple[int, int, int],
    batch_size: int,
    positive_probability: float = 0.5,
    seed: int = 42,
) -> DataLoader:
    """Build a reproducible DataLoader for sampled training patches."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero"
        )

    patch_dataset = TrainingPatchDataset(
        base_dataset=base_dataset,
        spatial_size=spatial_size,
        positive_probability=positive_probability,
        seed=seed,
    )

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    return DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=shuffle_generator,
    )

def build_brats_training_dataloader(
    cases_root: str | Path,
    manifest_path: str | Path,
    spatial_size: tuple[int, int, int],
    batch_size: int,
    positive_probability: float = 0.5,
    seed: int = 42,
) -> DataLoader:
    """Build the training DataLoader from the frozen BraTS train split."""

    base_dataset = BraTSDataset.from_manifest(
        cases_root=cases_root,
        manifest_path=manifest_path,
        split_name="train",
    )

    return build_training_dataloader(
        base_dataset=base_dataset,
        spatial_size=spatial_size,
        batch_size=batch_size,
        positive_probability=positive_probability,
        seed=seed,
    )

def build_brats_validation_dataloader(
    cases_root: str | Path,
    manifest_path: str | Path,
) -> DataLoader:
    """Build a deterministic DataLoader from the frozen BraTS validation split."""

    validation_dataset = BraTSDataset.from_manifest(
        cases_root=cases_root,
        manifest_path=manifest_path,
        split_name="validation",
    )

    return DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
