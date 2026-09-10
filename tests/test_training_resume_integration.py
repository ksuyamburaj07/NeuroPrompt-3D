from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.models.unet3d import LightweightUNet3D
from src.training.config import BaselineTrainingConfig
from src.training.workflow import run_baseline_training


class TinyTrainingDataset(Dataset):
    def __init__(self):
        self.generator = torch.Generator()
        self.generator.manual_seed(123)

        self.mri = torch.randn(
            4,
            16,
            16,
            16,
            generator=self.generator,
        )

        self.target = (
            torch.rand(
                16,
                16,
                16,
                generator=self.generator,
            )
            > 0.5
        ).float()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.mri, self.target


def build_tiny_experiment(
    num_epochs: int,
):
    config = BaselineTrainingConfig(
        patch_size=(16, 16, 16),
        batch_size=1,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        base_channels=2,
        dropout_probability=0.2,
        positive_probability=0.5,
        validation_overlap=0.25,
        seed=42,
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

    training_dataset = TinyTrainingDataset()

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(
        config.seed,
    )

    train_loader = DataLoader(
        training_dataset,
        batch_size=1,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
    )

    validation_mri = torch.randn(
        1,
        4,
        16,
        16,
        16,
    )

    validation_target = (
        torch.rand(
            1,
            16,
            16,
            16,
        )
        > 0.5
    ).float()

    validation_loader = DataLoader(
        TensorDataset(
            validation_mri,
            validation_target,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    return SimpleNamespace(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        device=torch.device("cpu"),
    )


def test_real_cpu_training_resumes_from_saved_checkpoint(
    tmp_path,
):
    output_dir = (
        tmp_path / "resume_integration"
    )

    first_experiment = build_tiny_experiment(
        num_epochs=1,
    )

    first_history = run_baseline_training(
        experiment=first_experiment,
        output_dir=output_dir,
        resume=False,
    )

    assert [
        result.epoch
        for result in first_history
    ] == [1]

    latest_checkpoint_path = (
        output_dir / "latest_baseline.pt"
    )

    history_path = (
        output_dir / "training_history.json"
    )

    assert latest_checkpoint_path.exists()
    assert history_path.exists()

    first_checkpoint = torch.load(
        latest_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert first_checkpoint["epoch"] == 1
    assert (
        first_checkpoint["training_random_state"]
        is not None
    )

    second_experiment = build_tiny_experiment(
        num_epochs=2,
    )

    resumed_history = run_baseline_training(
        experiment=second_experiment,
        output_dir=output_dir,
        resume=True,
    )

    assert [
        result.epoch
        for result in resumed_history
    ] == [
        1,
        2,
    ]

    final_checkpoint = torch.load(
        latest_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert final_checkpoint["epoch"] == 2
    assert (
        final_checkpoint["training_random_state"]
        is not None
    )
