import json
import os
from dataclasses import asdict
from pathlib import Path

from src.training.runner import EpochResult


def save_training_history(
    path: str | Path,
    history: list[EpochResult],
) -> None:
    """Save epoch-by-epoch training history as JSON."""
    history_path = Path(path)

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_history = [
        asdict(result)
        for result in history
    ]

    temporary_path = history_path.with_name(
        f".{history_path.name}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                serializable_history,
                file,
                indent=2,
        )


        os.replace(
            temporary_path,
            history_path,
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def load_training_history(
    path: str | Path,
) -> list[EpochResult]:
    """Load saved training history as EpochResult objects."""
    history_path = Path(path)

    with history_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        serialized_history = json.load(file)

    return [
        EpochResult(
            epoch=item["epoch"],
            train_loss=item["train_loss"],
            validation_loss=item["validation_loss"],
        )
        for item in serialized_history
    ]
