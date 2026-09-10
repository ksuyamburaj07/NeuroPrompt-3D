from types import SimpleNamespace
from unittest.mock import Mock

import torch

from src.training.resume import restore_baseline_training_state


def test_restore_baseline_training_state_loads_checkpoint_and_rng(
    monkeypatch,
    tmp_path,
):
    checkpoint = {
        "epoch": 4,
        "best_validation_loss": 0.8,
        "training_random_state": {
            "torch_cpu_rng_state": torch.tensor(
                [1, 2, 3],
                dtype=torch.uint8,
            ),
        },
    }

    load_checkpoint_mock = Mock(
        return_value=checkpoint,
    )

    restore_random_state_mock = Mock()

    monkeypatch.setattr(
        "src.training.resume.load_training_checkpoint",
        load_checkpoint_mock,
    )

    monkeypatch.setattr(
        "src.training.resume.restore_training_random_state",
        restore_random_state_mock,
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        device=torch.device("cpu"),
    )

    checkpoint_path = (
        tmp_path / "latest_baseline.pt"
    )

    returned_checkpoint = (
        restore_baseline_training_state(
            experiment=experiment,
            checkpoint_path=checkpoint_path,
        )
    )

    assert returned_checkpoint is checkpoint

    load_checkpoint_mock.assert_called_once_with(
        path=checkpoint_path,
        model=experiment.model,
        optimizer=experiment.optimizer,
        map_location=experiment.device,
    )

    restore_random_state_mock.assert_called_once_with(
        state=checkpoint["training_random_state"],
        train_loader=experiment.train_loader,
    )
