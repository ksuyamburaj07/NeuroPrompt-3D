import torch
from torch import nn

from src.training.config import BaselineTrainingConfig
from src.training.setup import (
    build_baseline_dataloaders,
    build_baseline_experiment,
    build_baseline_model,
    build_baseline_optimizer,
)


def test_build_baseline_model_uses_training_config():
    config = BaselineTrainingConfig(
        base_channels=4,
        dropout_probability=0.35,
    )

    model = build_baseline_model(config)

    first_conv = model.encoder1.block[0]

    assert isinstance(first_conv, nn.Conv3d)
    assert first_conv.in_channels == 4
    assert first_conv.out_channels == 4

    dropout_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.Dropout3d)
    ]

    assert len(dropout_layers) == 5

    assert all(
        layer.p == 0.35
        for layer in dropout_layers
    )

    assert model.output_conv.out_channels == 1


def test_build_baseline_optimizer_uses_config_learning_rate():
    config = BaselineTrainingConfig(
        learning_rate=5e-4,
    )

    model = build_baseline_model(config)

    optimizer = build_baseline_optimizer(
        model=model,
        config=config,
    )

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )

    assert optimizer.param_groups[0]["lr"] == 5e-4

def test_build_baseline_dataloaders_uses_config_and_frozen_manifest(
    tmp_path,
    monkeypatch,
):
    config = BaselineTrainingConfig(
        patch_size=(64, 64, 64),
        batch_size=2,
        positive_probability=0.7,
        seed=123,
    )

    captured = {}

    fake_train_loader = object()
    fake_validation_loader = object()

    def fake_build_training_loader(
        cases_root,
        manifest_path,
        spatial_size,
        batch_size,
        positive_probability,
        seed,
    ):
        captured["train"] = {
            "cases_root": cases_root,
            "manifest_path": manifest_path,
            "spatial_size": spatial_size,
            "batch_size": batch_size,
            "positive_probability": positive_probability,
            "seed": seed,
        }

        return fake_train_loader

    def fake_build_validation_loader(
        cases_root,
        manifest_path,
    ):
        captured["validation"] = {
            "cases_root": cases_root,
            "manifest_path": manifest_path,
        }

        return fake_validation_loader

    monkeypatch.setattr(
        "src.training.setup.build_brats_training_dataloader",
        fake_build_training_loader,
    )

    monkeypatch.setattr(
        "src.training.setup.build_brats_validation_dataloader",
        fake_build_validation_loader,
    )

    cases_root = tmp_path / "cases"
    manifest_path = tmp_path / "frozen_split.json"

    train_loader, validation_loader = build_baseline_dataloaders(
        cases_root=cases_root,
        manifest_path=manifest_path,
        config=config,
    )

    assert train_loader is fake_train_loader
    assert validation_loader is fake_validation_loader

    assert captured["train"] == {
        "cases_root": cases_root,
        "manifest_path": manifest_path,
        "spatial_size": (64, 64, 64),
        "batch_size": 2,
        "positive_probability": 0.7,
        "seed": 123,
    }

    assert captured["validation"] == {
        "cases_root": cases_root,
        "manifest_path": manifest_path,
    }

def test_build_baseline_experiment_returns_complete_training_setup(
    tmp_path,
    monkeypatch,
):
    config = BaselineTrainingConfig(
        base_channels=4,
        dropout_probability=0.3,
        learning_rate=5e-4,
    )

    fake_train_loader = object()
    fake_validation_loader = object()

    def fake_build_dataloaders(
        cases_root,
        manifest_path,
        config,
    ):
        return (
            fake_train_loader,
            fake_validation_loader,
        )

    monkeypatch.setattr(
        "src.training.setup.build_baseline_dataloaders",
        fake_build_dataloaders,
    )

    experiment = build_baseline_experiment(
        cases_root=tmp_path / "cases",
        manifest_path=tmp_path / "frozen_split.json",
        config=config,
    )

    assert experiment.model.encoder1.block[0].out_channels == 4

    assert experiment.optimizer.param_groups[0]["lr"] == 5e-4

    assert experiment.train_loader is fake_train_loader
    assert experiment.validation_loader is fake_validation_loader

    assert experiment.config is config

def test_build_baseline_experiment_replays_model_initialization_with_same_seed(
    tmp_path,
    monkeypatch,
):
    fake_train_loader = object()
    fake_validation_loader = object()

    def fake_build_dataloaders(
        cases_root,
        manifest_path,
        config,
    ):
        return (
            fake_train_loader,
            fake_validation_loader,
        )

    monkeypatch.setattr(
        "src.training.setup.build_baseline_dataloaders",
        fake_build_dataloaders,
    )

    config_a = BaselineTrainingConfig(
        seed=123,
        base_channels=4,
    )

    config_b = BaselineTrainingConfig(
        seed=123,
        base_channels=4,
    )

    experiment_a = build_baseline_experiment(
        cases_root=tmp_path / "cases",
        manifest_path=tmp_path / "frozen_split.json",
        config=config_a,
    )

    experiment_b = build_baseline_experiment(
        cases_root=tmp_path / "cases",
        manifest_path=tmp_path / "frozen_split.json",
        config=config_b,
    )

    state_a = experiment_a.model.state_dict()
    state_b = experiment_b.model.state_dict()

    assert state_a.keys() == state_b.keys()

    assert all(
        torch.equal(
            state_a[name],
            state_b[name],
        )
        for name in state_a
    )

def test_build_baseline_experiment_resolves_and_uses_config_device(
    tmp_path,
    monkeypatch,
):
    fake_train_loader = object()
    fake_validation_loader = object()

    def fake_build_dataloaders(
        cases_root,
        manifest_path,
        config,
    ):
        return (
            fake_train_loader,
            fake_validation_loader,
        )

    captured = {}

    def fake_resolve_device(
        requested_device,
    ):
        captured["requested_device"] = requested_device
        return torch.device("cpu")

    monkeypatch.setattr(
        "src.training.setup.build_baseline_dataloaders",
        fake_build_dataloaders,
    )

    monkeypatch.setattr(
        "src.training.setup.resolve_device",
        fake_resolve_device,
    )

    config = BaselineTrainingConfig(
        device="cpu",
        base_channels=4,
    )

    experiment = build_baseline_experiment(
        cases_root=tmp_path / "cases",
        manifest_path=tmp_path / "frozen_split.json",
        config=config,
    )

    assert captured["requested_device"] == "cpu"

    assert experiment.device == torch.device("cpu")

    assert (
        next(experiment.model.parameters()).device
        == experiment.device
    )
