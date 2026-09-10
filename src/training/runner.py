from dataclasses import dataclass
from pathlib import Path

from src.training.checkpoint import save_training_checkpoint

from src.training.engine import (
    evaluate_sliding_window_epoch,
    train_one_epoch,
)


@dataclass(frozen=True)
class EpochResult:
    """Loss summary for one completed training epoch."""

    epoch: int
    train_loss: float
    validation_loss: float


def run_baseline_epochs(
    experiment,
    num_epochs: int,
    checkpoint_path: str | Path | None = None,
    start_epoch: int = 1,
    initial_best_validation_loss: float = float("inf"),
    on_epoch_complete=None,
) -> list[EpochResult]:
    """Run baseline training and validation for multiple epochs."""

    if num_epochs <= 0:
        raise ValueError(
            "num_epochs must be greater than zero"
        )

    history: list[EpochResult] = []
    best_validation_loss = (
        initial_best_validation_loss
    )

    for epoch in range(start_epoch, num_epochs + 1):
        print(
            f"Epoch {epoch}/{num_epochs} started",
            flush=True,
        )

        train_loss = train_one_epoch(
            model=experiment.model,
            optimizer=experiment.optimizer,
            dataloader=experiment.train_loader,
        )

        validation_loss = evaluate_sliding_window_epoch(
            model=experiment.model,
            dataloader=experiment.validation_loader,
            roi_size=experiment.config.patch_size,
            overlap=experiment.config.validation_overlap,
        )

        print(
            f"Epoch {epoch}/{num_epochs} complete - "
            f"train_loss={train_loss:.6f} - "
            f"validation_loss={validation_loss:.6f}",
            flush=True,
        )

        history.append(
            EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
            )
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            if checkpoint_path is not None:
                save_training_checkpoint(
                    path=checkpoint_path,
                    model=experiment.model,
                    optimizer=experiment.optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    config=experiment.config,
                )

        if on_epoch_complete is not None:
            on_epoch_complete(
                list(history),
                best_validation_loss,
            )

    return history
