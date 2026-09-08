import json

import torch

from src.models.unet3d import LightweightUNet3D
from src.training.config import BaselineTrainingConfig

from types import SimpleNamespace
from unittest.mock import Mock

from src.training.runner import EpochResult
from src.training.workflow import run_baseline_training


def test_run_baseline_training_saves_checkpoint_and_history(
    tmp_path,
    monkeypatch,
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

    run_epochs_mock = Mock(
        return_value=history,
    )

    save_history_mock = Mock()

    monkeypatch.setattr(
        "src.training.workflow.run_baseline_epochs",
        run_epochs_mock,
    )

    monkeypatch.setattr(
        "src.training.workflow.save_training_history",
        save_history_mock,
    )

    experiment = SimpleNamespace()

    returned_history = run_baseline_training(
        experiment=experiment,
        num_epochs=2,
        output_dir=tmp_path,
    )

    assert returned_history == history

    run_epochs_mock.assert_called_once_with(
        experiment=experiment,
        num_epochs=2,
        checkpoint_path=(
            tmp_path / "best_baseline.pt"
        ),
    )

    save_history_mock.assert_called_once_with(
        path=(
            tmp_path / "training_history.json"
        ),
        history=history,
    )

def test_run_baseline_training_creates_real_output_files(
    tmp_path,
    monkeypatch,
):
    train_losses = iter([
        1.4,
        1.2,
    ])

    validation_losses = iter([
        1.5,
        1.3,
    ])

    def fake_train_one_epoch(
        model,
        optimizer,
        dataloader,
    ):
        return next(train_losses)

    def fake_evaluate_sliding_window_epoch(
        model,
        dataloader,
        roi_size,
        overlap,
    ):
        return next(validation_losses)

    monkeypatch.setattr(
        "src.training.runner.train_one_epoch",
        fake_train_one_epoch,
    )

    monkeypatch.setattr(
        "src.training.runner.evaluate_sliding_window_epoch",
        fake_evaluate_sliding_window_epoch,
    )

    config = BaselineTrainingConfig(
        base_channels=2,
        device="cpu",
    )

    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=config.base_channels,
        dropout_probability=config.dropout_probability,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    experiment = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        train_loader=object(),
        validation_loader=object(),
        config=config,
    )

    output_dir = tmp_path / "baseline_run"

    history = run_baseline_training(
        experiment=experiment,
        num_epochs=2,
        output_dir=output_dir,
    )

    checkpoint_path = (
        output_dir / "best_baseline.pt"
    )

    history_path = (
        output_dir / "training_history.json"
    )

    assert checkpoint_path.exists()
    assert history_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["epoch"] == 2
    assert checkpoint["train_loss"] == 1.2
    assert checkpoint["validation_loss"] == 1.3

    with history_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_history = json.load(file)

    assert len(history) == 2
    assert len(saved_history) == 2

    assert saved_history[-1] == {
        "epoch": 2,
        "train_loss": 1.2,
        "validation_loss": 1.3,
    }
