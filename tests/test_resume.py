from types import SimpleNamespace
from unittest.mock import Mock

import torch
import pytest

from src.training.config import BaselineTrainingConfig
from src.training.resume import (
    restore_baseline_training_state,
    validate_resume_configuration,
)

def test_restore_baseline_training_state_loads_checkpoint_and_rng(
    monkeypatch,
    tmp_path,
):
    config = BaselineTrainingConfig(
        base_channels=8,
    )

    checkpoint = {
        "epoch": 4,
        "best_validation_loss": 0.8,
        "config": vars(config),
        "training_random_state": {
            "torch_cpu_rng_state": torch.tensor(
                [1, 2, 3],
                dtype=torch.uint8,
            ),
        },
    }

    read_checkpoint_mock = Mock(
        return_value=checkpoint,
    )
    restore_checkpoint_state_mock = Mock()
    restore_random_state_mock = Mock()

    monkeypatch.setattr(
        "src.training.resume.read_training_checkpoint",
        read_checkpoint_mock,
    )
    monkeypatch.setattr(
        "src.training.resume.restore_training_checkpoint_state",
        restore_checkpoint_state_mock,
    )
    monkeypatch.setattr(
        "src.training.resume.restore_training_random_state",
        restore_random_state_mock,
    )

    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        config=config,
    )

    checkpoint_path = (
        tmp_path / "latest_baseline.pt"
    )

    restored_checkpoint = (
        restore_baseline_training_state(
            experiment=experiment,
            checkpoint_path=checkpoint_path,
        )
    )

    read_checkpoint_mock.assert_called_once_with(
        path=checkpoint_path,
        map_location="cpu",
    )

    restore_checkpoint_state_mock.assert_called_once_with(
        checkpoint=checkpoint,
        model=experiment.model,
        optimizer=experiment.optimizer,
    )

    restore_random_state_mock.assert_called_once_with(
        state=checkpoint["training_random_state"],
        train_loader=experiment.train_loader,
    )

    assert restored_checkpoint is checkpoint

def test_validate_resume_configuration_allows_epoch_and_device_changes():
    checkpoint_config = BaselineTrainingConfig(
        base_channels=8,
        learning_rate=0.001,
        num_epochs=5,
        device="cuda",
    )

    current_config = BaselineTrainingConfig(
        base_channels=8,
        learning_rate=0.001,
        num_epochs=10,
        device="cpu",
    )

    validate_resume_configuration(
        checkpoint_config=vars(checkpoint_config),
        current_config=current_config,
    )


def test_validate_resume_configuration_rejects_training_setting_change():
    checkpoint_config = BaselineTrainingConfig(
        base_channels=8,
        learning_rate=0.001,
        num_epochs=5,
        device="cuda",
    )

    current_config = BaselineTrainingConfig(
        base_channels=16,
        learning_rate=0.001,
        num_epochs=10,
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="base_channels",
    ):
        validate_resume_configuration(
            checkpoint_config=vars(checkpoint_config),
            current_config=current_config,
        )

def test_restore_validates_configuration_before_restoring_random_state(
    monkeypatch,
):
    experiment = SimpleNamespace(
        model=object(),
        optimizer=object(),
        train_loader=object(),
        config=BaselineTrainingConfig(
            base_channels=16,
        ),
    )

    checkpoint = {
        "config": vars(
            BaselineTrainingConfig(
                base_channels=8,
            )
        ),
        "training_random_state": {
            "example": "state",
        },
    }

    read_checkpoint_mock = Mock(
        return_value=checkpoint,
    )

    restore_checkpoint_state_mock = Mock()
    restore_random_state_mock = Mock()

    monkeypatch.setattr(
        "src.training.resume.read_training_checkpoint",
        read_checkpoint_mock,
    )

    monkeypatch.setattr(
        "src.training.resume.restore_training_checkpoint_state",
        restore_checkpoint_state_mock,
    )

    monkeypatch.setattr(
        "src.training.resume.restore_training_random_state",
        restore_random_state_mock,
    )

    with pytest.raises(
        ValueError,
        match="base_channels",
    ):
        restore_baseline_training_state(
            experiment=experiment,
            checkpoint_path="latest_baseline.pt",
        )

    restore_checkpoint_state_mock.assert_not_called()
    restore_random_state_mock.assert_not_called()
