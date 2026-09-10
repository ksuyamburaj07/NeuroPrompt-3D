from pathlib import Path

from src.training.checkpoint import (
    save_training_checkpoint,
)
from src.training.history import (
    load_training_history,
    save_training_history,
)
from src.training.random_state import (
    capture_training_random_state,
)
from src.training.resume import (
    restore_baseline_training_state,
)
from src.training.runner import (
    EpochResult,
    run_baseline_epochs,
)


def run_baseline_training(
    experiment,
    output_dir: str | Path,
    resume: bool = False,
) -> list[EpochResult]:
    """Run baseline training, optionally resuming from saved state."""
    output_path = Path(output_dir)

    latest_checkpoint_path = (
        output_path / "latest_baseline.pt"
    )

    history_path = (
        output_path / "training_history.json"
    )

    previous_history: list[EpochResult] = []
    start_epoch = 1
    initial_best_validation_loss = float("inf")

    if resume:
        if not latest_checkpoint_path.is_file():
            raise FileNotFoundError(
                "Cannot resume training because the latest "
                f"checkpoint does not exist: {latest_checkpoint_path}"
            )

        if not history_path.is_file():
            raise FileNotFoundError(
                "Cannot resume training because the history "
                f"file does not exist: {history_path}"
            )

        checkpoint = restore_baseline_training_state(
            experiment=experiment,
            checkpoint_path=latest_checkpoint_path,
        )

        previous_history = load_training_history(
            path=history_path,
        )

        history_last_epoch = (
            previous_history[-1].epoch
            if previous_history
            else 0
        )

        if history_last_epoch == checkpoint["epoch"] + 1:
            previous_history = previous_history[:-1]

            save_training_history(
                path=history_path,
                history=previous_history,
            )

            history_last_epoch = (
                previous_history[-1].epoch
                if previous_history
                else 0
            )

        if history_last_epoch != checkpoint["epoch"]:
            raise ValueError(
                "Cannot resume training because checkpoint epoch "
                "does not match training history: "
                f"checkpoint epoch {checkpoint['epoch']}, "
                f"history last epoch {history_last_epoch}."
            )

        start_epoch = checkpoint["epoch"] + 1

        initial_best_validation_loss = checkpoint[
            "best_validation_loss"
        ]

    def persist_completed_epoch(
        current_history: list[EpochResult],
        best_validation_loss: float,
    ) -> None:
        latest_result = current_history[-1]

        combined_history = [
            *previous_history,
            *current_history,
        ]

        random_state = capture_training_random_state(
            train_loader=experiment.train_loader,
        )

        save_training_history(
            path=history_path,
            history=combined_history,
        )

        save_training_checkpoint(
            path=latest_checkpoint_path,
            model=experiment.model,
            optimizer=experiment.optimizer,
            epoch=latest_result.epoch,
            train_loss=latest_result.train_loss,
            validation_loss=latest_result.validation_loss,
            best_validation_loss=best_validation_loss,
            config=experiment.config,
            random_state=random_state,
        )

    if resume:
        current_history = run_baseline_epochs(
            experiment=experiment,
            num_epochs=experiment.config.num_epochs,
            checkpoint_path=(
                output_path / "best_baseline.pt"
            ),
            start_epoch=start_epoch,
            initial_best_validation_loss=(
                initial_best_validation_loss
            ),
            on_epoch_complete=persist_completed_epoch,
        )
    else:
        current_history = run_baseline_epochs(
            experiment=experiment,
            num_epochs=experiment.config.num_epochs,
            checkpoint_path=(
                output_path / "best_baseline.pt"
            ),
            on_epoch_complete=persist_completed_epoch,
        )

    return [
        *previous_history,
        *current_history,
    ]
