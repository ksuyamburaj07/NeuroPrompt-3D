import json

import torch

from src.models.unet3d import LightweightUNet3D
from src.training.config import BaselineTrainingConfig

from types import SimpleNamespace
from unittest.mock import Mock

from src.training.runner import EpochResult
from src.training.workflow import run_baseline_training


def test_run_baseline_training_configures_epoch_persistence_callback(
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

    experiment = SimpleNamespace(
    config=SimpleNamespace(
        num_epochs=2,
    ),
)

    returned_history = run_baseline_training(
        experiment=experiment,
        output_dir=tmp_path,
    )

    assert returned_history == history

    run_epochs_mock.assert_called_once()

    call_kwargs = run_epochs_mock.call_args.kwargs

    assert call_kwargs["experiment"] is experiment
    assert call_kwargs["num_epochs"] == 2

    assert call_kwargs["checkpoint_path"] == (
        tmp_path / "best_baseline.pt"
    )

    assert callable(
        call_kwargs["on_epoch_complete"]
    )

    save_history_mock.assert_not_called()

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

    random_state = {
        "torch_cpu_rng_state": torch.tensor(
            [1, 2, 3],
            dtype=torch.uint8,
        ),
        "torch_cuda_rng_state_all": None,
        "shuffle_generator_state": torch.tensor(
            [4, 5, 6],
            dtype=torch.uint8,
        ),
        "patch_generator_state": torch.tensor(
            [7, 8, 9],
            dtype=torch.uint8,
        ),
    }

    monkeypatch.setattr(
        "src.training.workflow.capture_training_random_state",
        lambda train_loader: random_state,
    )

    config = BaselineTrainingConfig(
        base_channels=2,
        device="cpu",
        num_epochs=2,
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
        output_dir=output_dir,
    )

    best_checkpoint_path = (
        output_dir / "best_baseline.pt"
    )

    latest_checkpoint_path = (
        output_dir / "latest_baseline.pt"
    )

    history_path = (
        output_dir / "training_history.json"
    )

    assert best_checkpoint_path.exists()
    assert latest_checkpoint_path.exists()
    assert history_path.exists()

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    latest_checkpoint = torch.load(
        latest_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert best_checkpoint["epoch"] == 2
    assert best_checkpoint["train_loss"] == 1.2
    assert best_checkpoint["validation_loss"] == 1.3

    assert latest_checkpoint["epoch"] == 2
    assert latest_checkpoint["train_loss"] == 1.2
    assert latest_checkpoint["validation_loss"] == 1.3
    assert (
        latest_checkpoint["best_validation_loss"]
        == 1.3
    )

    saved_random_state = (
        latest_checkpoint["training_random_state"]
    )

    assert torch.equal(
        saved_random_state["torch_cpu_rng_state"],
        random_state["torch_cpu_rng_state"],
    )

    assert (
        saved_random_state["torch_cuda_rng_state_all"]
        is None
    )

    assert torch.equal(
        saved_random_state["shuffle_generator_state"],
        random_state["shuffle_generator_state"],
    )

    assert torch.equal(
        saved_random_state["patch_generator_state"],
        random_state["patch_generator_state"],
    )

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

def test_run_baseline_training_persists_latest_state_after_each_epoch(
    tmp_path,
    monkeypatch,
):
    history_epoch_1 = [
        EpochResult(
            epoch=1,
            train_loss=1.4,
            validation_loss=1.5,
        ),
    ]

    history_epoch_2 = [
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

    def fake_run_baseline_epochs(
        experiment,
        num_epochs,
        checkpoint_path,
        on_epoch_complete,
    ):
        on_epoch_complete(
            history_epoch_1,
            1.5,
        )

        on_epoch_complete(
            history_epoch_2,
            1.3,
        )

        return history_epoch_2

    save_checkpoint_mock = Mock()
    save_history_mock = Mock()

    random_state = {
        "example": "state",
    }

    capture_random_state_mock = Mock(
        return_value=random_state,
    )

    monkeypatch.setattr(
        "src.training.workflow.run_baseline_epochs",
        fake_run_baseline_epochs,
    )

    monkeypatch.setattr(
        "src.training.workflow.save_training_checkpoint",
        save_checkpoint_mock,
    )

    monkeypatch.setattr(
        "src.training.workflow.save_training_history",
        save_history_mock,
    )

    monkeypatch.setattr(
        "src.training.workflow.capture_training_random_state",
        capture_random_state_mock,
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        config=SimpleNamespace(
            num_epochs=2,
        ),
    )

    returned_history = run_baseline_training(
        experiment=experiment,
        output_dir=tmp_path,
    )

    assert returned_history == history_epoch_2

    assert capture_random_state_mock.call_count == 2

    assert save_checkpoint_mock.call_count == 2

    first_checkpoint_call = (
        save_checkpoint_mock.call_args_list[0]
    )

    second_checkpoint_call = (
        save_checkpoint_mock.call_args_list[1]
    )

    assert first_checkpoint_call.kwargs[
        "path"
    ] == (
        tmp_path / "latest_baseline.pt"
    )

    assert first_checkpoint_call.kwargs[
        "epoch"
    ] == 1

    assert first_checkpoint_call.kwargs[
        "train_loss"
    ] == 1.4

    assert first_checkpoint_call.kwargs[
        "validation_loss"
    ] == 1.5

    assert first_checkpoint_call.kwargs[
        "best_validation_loss"
    ] == 1.5

    assert first_checkpoint_call.kwargs[
        "random_state"
    ] is random_state

    assert second_checkpoint_call.kwargs[
        "epoch"
    ] == 2

    assert second_checkpoint_call.kwargs[
        "best_validation_loss"
    ] == 1.3

    assert save_history_mock.call_count == 2

    assert (
        save_history_mock.call_args_list[0]
        .kwargs["history"]
        == history_epoch_1
    )

    assert (
        save_history_mock.call_args_list[1]
        .kwargs["history"]
        == history_epoch_2
    )

def test_run_baseline_training_saves_history_before_latest_checkpoint(
    tmp_path,
    monkeypatch,
):
    events = []

    epoch_result = EpochResult(
        epoch=1,
        train_loss=1.2,
        validation_loss=1.3,
    )

    def fake_run_baseline_epochs(
        experiment,
        num_epochs,
        checkpoint_path,
        on_epoch_complete,
    ):
        on_epoch_complete(
            [epoch_result],
            1.3,
        )

        return [epoch_result]

    def fake_save_history(
        path,
        history,
    ):
        events.append("history")

    def fake_save_checkpoint(**kwargs):
        events.append("checkpoint")

    monkeypatch.setattr(
        "src.training.workflow.run_baseline_epochs",
        fake_run_baseline_epochs,
    )

    monkeypatch.setattr(
        "src.training.workflow.save_training_history",
        fake_save_history,
    )

    monkeypatch.setattr(
        "src.training.workflow.save_training_checkpoint",
        fake_save_checkpoint,
    )

    monkeypatch.setattr(
        "src.training.workflow.capture_training_random_state",
        lambda train_loader: {},
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        config=SimpleNamespace(
            num_epochs=1,
        ),
    )

    run_baseline_training(
        experiment=experiment,
        output_dir=tmp_path,
    )

    assert events == [
        "history",
        "checkpoint",
    ]
