import json
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

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_history,
            file,
            indent=2,
        )
