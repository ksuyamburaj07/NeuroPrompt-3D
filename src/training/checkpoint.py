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
) -> None:
    """Save model, optimizer, metrics, and configuration."""

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )
