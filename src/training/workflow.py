from pathlib import Path

from src.training.checkpoint import (
    save_training_checkpoint,
)
from src.training.history import (
    save_training_history,
)
from src.training.random_state import (
    capture_training_random_state,
)
from src.training.runner import (
    EpochResult,
    run_baseline_epochs,
)


def run_baseline_training(
    experiment,
    output_dir: str | Path,
) -> list[EpochResult]:
    """Run baseline training and persist each completed epoch."""
    output_path = Path(output_dir)

    def persist_completed_epoch(
        history: list[EpochResult],
        best_validation_loss: float,
    ) -> None:
        latest_result = history[-1]

        random_state = capture_training_random_state(
            train_loader=experiment.train_loader,
        )

        save_training_checkpoint(
            path=(
                output_path / "latest_baseline.pt"
            ),
            model=experiment.model,
            optimizer=experiment.optimizer,
            epoch=latest_result.epoch,
            train_loss=latest_result.train_loss,
            validation_loss=(
                latest_result.validation_loss
            ),
            best_validation_loss=(
                best_validation_loss
            ),
            config=experiment.config,
            random_state=random_state,
        )

        save_training_history(
            path=(
                output_path
                / "training_history.json"
            ),
            history=history,
        )

    history = run_baseline_epochs(
        experiment=experiment,
        num_epochs=experiment.config.num_epochs,
        checkpoint_path=(
            output_path / "best_baseline.pt"
        ),
        on_epoch_complete=persist_completed_epoch,
    )

    return history
