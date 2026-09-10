from dataclasses import asdict
from pathlib import Path

import torch

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

    torch.save(
        checkpoint,
        checkpoint_path,
    )

def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a training checkpoint and restore model and optimizer state."""
    checkpoint_path = Path(path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return checkpoint
