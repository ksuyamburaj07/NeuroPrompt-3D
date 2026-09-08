import torch

from src.models.unet3d import LightweightUNet3D
from src.training.checkpoint import save_training_checkpoint
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
