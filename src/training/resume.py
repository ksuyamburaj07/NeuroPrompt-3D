from pathlib import Path

from src.training.checkpoint import load_training_checkpoint
from src.training.random_state import restore_training_random_state


def restore_baseline_training_state(
    experiment,
    checkpoint_path: str | Path,
) -> dict:
    """Restore a baseline experiment from a saved training checkpoint."""
    checkpoint = load_training_checkpoint(
        path=checkpoint_path,
        model=experiment.model,
        optimizer=experiment.optimizer,
        map_location=experiment.device,
    )

    restore_training_random_state(
        state=checkpoint["training_random_state"],
        train_loader=experiment.train_loader,
    )

    return checkpoint
