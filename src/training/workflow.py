from pathlib import Path

from src.training.history import save_training_history
from src.training.runner import (
    EpochResult,
    run_baseline_epochs,
)


def run_baseline_training(
    experiment,
    num_epochs: int,
    output_dir: str | Path,
) -> list[EpochResult]:
    """Run baseline training and persist its main outputs."""

    output_path = Path(output_dir)

    history = run_baseline_epochs(
        experiment=experiment,
        num_epochs=num_epochs,
        checkpoint_path=(
            output_path / "best_baseline.pt"
        ),
    )

    save_training_history(
        path=(
            output_path / "training_history.json"
        ),
        history=history,
    )

    return history
