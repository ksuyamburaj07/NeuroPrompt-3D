import json

from src.training.history import save_training_history
from src.training.runner import EpochResult


def test_save_training_history_writes_epoch_results_to_json(
    tmp_path,
):
    history = [
        EpochResult(
            epoch=1,
            train_loss=1.4,
            validation_loss=1.5,
        ),
        EpochResult(
            epoch=2,
            train_loss=1.2,
            validation_loss=1.3,
        ),
    ]

    history_path = (
        tmp_path
        / "results"
        / "training_history.json"
    )

    save_training_history(
        path=history_path,
        history=history,
    )

    assert history_path.exists()

    with history_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_history = json.load(file)

    assert saved_history == [
        {
            "epoch": 1,
            "train_loss": 1.4,
            "validation_loss": 1.5,
        },
        {
            "epoch": 2,
            "train_loss": 1.2,
            "validation_loss": 1.3,
        },
    ]
