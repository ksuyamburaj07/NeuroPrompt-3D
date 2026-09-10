from pathlib import Path

import pytest
import json

from src.training.history import (
    load_training_history,
    save_training_history,
)
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

def test_load_training_history_restores_epoch_results(
    tmp_path,
):
    history_path = (
        tmp_path / "training_history.json"
    )

    history_path.write_text(
        """
[
  {
    "epoch": 1,
    "train_loss": 1.4,
    "validation_loss": 1.5
  },
  {
    "epoch": 2,
    "train_loss": 1.2,
    "validation_loss": 1.3
  }
]
""".strip(),
        encoding="utf-8",
    )

    history = load_training_history(
        path=history_path,
    )

    assert len(history) == 2

    assert history[0].epoch == 1
    assert history[0].train_loss == 1.4
    assert history[0].validation_loss == 1.5

    assert history[1].epoch == 2
    assert history[1].train_loss == 1.2
    assert history[1].validation_loss == 1.3

def test_save_training_history_preserves_existing_file_if_write_fails(
    tmp_path,
    monkeypatch,
):
    history_path = (
        tmp_path / "training_history.json"
    )

    history_path.write_text(
        "previous-valid-history",
        encoding="utf-8",
    )

    history = [
        EpochResult(
            epoch=1,
            train_loss=1.4,
            validation_loss=1.5,
        ),
    ]

    def failing_json_dump(
        obj,
        file,
        *args,
        **kwargs,
    ):
        file.write(
            "incomplete-new-history"
        )

        raise RuntimeError(
            "simulated interrupted history save"
        )

    monkeypatch.setattr(
        "src.training.history.json.dump",
        failing_json_dump,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated interrupted history save",
    ):
        save_training_history(
            path=history_path,
            history=history,
        )

    assert history_path.read_text(
        encoding="utf-8",
    ) == "previous-valid-history"
