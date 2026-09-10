from dataclasses import asdict
from pathlib import Path

import torch
import os
from src.training.config import BaselineTrainingConfig


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    config: BaselineTrainingConfig,
    best_validation_loss: float | None = None,
    random_state: dict[str, object] | None = None,
) -> None:
    """Save model, optimizer, metrics, configuration, and resume state."""

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_random_state": random_state,
    }

    temporary_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.tmp"
    )

    try:
        torch.save(
            checkpoint,
            temporary_path,
        )

        os.replace(
            temporary_path,
            checkpoint_path,
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def read_training_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Read a trusted training checkpoint without restoring state."""
    checkpoint_path = Path(path)

    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )


def restore_training_checkpoint_state(
    checkpoint: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Restore model and optimizer state from a loaded checkpoint."""
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint and restore model and optimizer state."""
    checkpoint = read_training_checkpoint(
        path=path,
        map_location=map_location,
    )

    restore_training_checkpoint_state(
        checkpoint=checkpoint,
        model=model,
        optimizer=optimizer,
    )

    return checkpoint
