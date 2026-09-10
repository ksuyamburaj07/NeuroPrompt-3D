from pathlib import Path

import pytest
import torch

from src.models.unet3d import LightweightUNet3D
from src.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.training.config import BaselineTrainingConfig


def test_save_training_checkpoint_preserves_training_state(
    tmp_path,
):
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

    checkpoint_path = tmp_path / "best_baseline.pt"

    save_training_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        train_loss=1.2,
        validation_loss=1.1,
        config=config,
    )

    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["train_loss"] == 1.2
    assert checkpoint["validation_loss"] == 1.1

    assert checkpoint["config"]["base_channels"] == 2
    assert checkpoint["config"]["device"] == "cpu"

    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint

def test_save_training_checkpoint_preserves_resume_state(
    tmp_path,
):
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

    checkpoint_path = (
        tmp_path / "latest_baseline.pt"
    )

    save_training_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=4,
        train_loss=1.0,
        validation_loss=0.9,
        best_validation_loss=0.8,
        config=config,
        random_state=random_state,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint[
        "best_validation_loss"
    ] == 0.8

    saved_random_state = checkpoint[
        "training_random_state"
    ]

    assert torch.equal(
        saved_random_state[
            "torch_cpu_rng_state"
        ],
        random_state[
            "torch_cpu_rng_state"
        ],
    )

    assert (
        saved_random_state[
            "torch_cuda_rng_state_all"
        ]
        is None
    )

    assert torch.equal(
        saved_random_state[
            "shuffle_generator_state"
        ],
        random_state[
            "shuffle_generator_state"
        ],
    )

    assert torch.equal(
        saved_random_state[
            "patch_generator_state"
        ],
        random_state[
            "patch_generator_state"
        ],
    )

def test_load_training_checkpoint_restores_model_optimizer_and_metadata(
    tmp_path,
):
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

    original_model_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

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

    checkpoint_path = (
        tmp_path / "latest_baseline.pt"
    )

    save_training_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=4,
        train_loss=1.0,
        validation_loss=0.9,
        best_validation_loss=0.8,
        config=config,
        random_state=random_state,
    )

    # Deliberately change the live training state.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    optimizer.param_groups[0]["lr"] = 0.5

    checkpoint = load_training_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        map_location="cpu",
    )

    for key, value in model.state_dict().items():
        assert torch.equal(
            value,
            original_model_state[key],
        )

    assert (
        optimizer.param_groups[0]["lr"]
        == config.learning_rate
    )

    assert checkpoint["epoch"] == 4
    assert checkpoint["train_loss"] == 1.0
    assert checkpoint["validation_loss"] == 0.9
    assert checkpoint["best_validation_loss"] == 0.8

    assert (
        checkpoint["training_random_state"]
        is not None
    )

def test_save_training_checkpoint_preserves_existing_file_if_save_fails(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = (
        tmp_path / "latest_baseline.pt"
    )

    checkpoint_path.write_bytes(
        b"previous-valid-checkpoint"
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

    def failing_torch_save(payload, path):
        Path(path).write_bytes(
            b"incomplete-new-checkpoint"
        )

        raise RuntimeError(
            "simulated interrupted save"
        )

    monkeypatch.setattr(
        torch,
        "save",
        failing_torch_save,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated interrupted save",
    ):
        save_training_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=3,
            train_loss=1.0,
            validation_loss=1.1,
            config=config,
        )

    assert checkpoint_path.read_bytes() == (
        b"previous-valid-checkpoint"
    )
