import torch
import math

from torch.utils.data import DataLoader, TensorDataset
from src.models.unet3d import LightweightUNet3D
from src.training.engine import (
    evaluate_one_batch,
    evaluate_one_epoch,
    train_one_batch,
    train_one_epoch,
)


def test_train_one_batch_updates_model_parameters():
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=4,
        dropout_probability=0.1,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    mri = torch.randn(
        1,
        4,
        16,
        16,
        16,
    )

    target = torch.zeros(
        (1, 16, 16, 16),
        dtype=torch.uint8,
    )

    target[
        :,
        4:12,
        4:12,
        4:12,
    ] = 1

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    loss = train_one_batch(
        model=model,
        optimizer=optimizer,
        mri=mri,
        target=target,
    )

    after = dict(model.named_parameters())

    assert torch.isfinite(loss)

    assert any(
        not torch.equal(
            before[name],
            after[name].detach(),
        )
        for name in before
    )

def test_evaluate_one_batch_does_not_update_model_parameters():
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=4,
        dropout_probability=0.1,
    )

    mri = torch.randn(
        1,
        4,
        16,
        16,
        16,
    )

    target = torch.zeros(
        (1, 16, 16, 16),
        dtype=torch.uint8,
    )

    target[
        :,
        4:12,
        4:12,
        4:12,
    ] = 1

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    loss = evaluate_one_batch(
        model=model,
        mri=mri,
        target=target,
    )

    after = dict(model.named_parameters())

    assert torch.isfinite(loss)
    assert model.training is False

    assert all(
        torch.equal(
            before[name],
            after[name].detach(),
        )
        for name in before
    )

def test_model_can_overfit_one_easy_synthetic_batch():
    torch.manual_seed(42)

    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=2,
        dropout_probability=0.0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
    )

    mri = torch.zeros(
        (1, 4, 16, 16, 16),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (1, 16, 16, 16),
        dtype=torch.uint8,
    )

    target[
        :,
        4:12,
        4:12,
        4:12,
    ] = 1

    mri[
        :,
        0,
        4:12,
        4:12,
        4:12,
    ] = 1.0

    initial_loss = evaluate_one_batch(
        model=model,
        mri=mri,
        target=target,
    )

    for _ in range(20):
        train_one_batch(
            model=model,
            optimizer=optimizer,
            mri=mri,
            target=target,
        )

    final_loss = evaluate_one_batch(
        model=model,
        mri=mri,
        target=target,
    )

    assert final_loss < initial_loss

def test_train_one_epoch_processes_dataloader_and_returns_mean_loss():
    torch.manual_seed(42)

    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=2,
        dropout_probability=0.0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    mri = torch.zeros(
        (2, 4, 16, 16, 16),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (2, 16, 16, 16),
        dtype=torch.uint8,
    )

    target[
        :,
        4:12,
        4:12,
        4:12,
    ] = 1

    mri[
        :,
        0,
        4:12,
        4:12,
        4:12,
    ] = 1.0

    dataset = TensorDataset(
        mri,
        target,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    epoch_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        dataloader=loader,
    )

    assert isinstance(epoch_loss, float)
    assert math.isfinite(epoch_loss)
    assert epoch_loss >= 0.0

def test_evaluate_one_epoch_returns_mean_loss_without_updating_parameters():
    torch.manual_seed(42)

    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=2,
        dropout_probability=0.1,
    )

    mri = torch.zeros(
        (2, 4, 16, 16, 16),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (2, 16, 16, 16),
        dtype=torch.uint8,
    )

    target[
        :,
        4:12,
        4:12,
        4:12,
    ] = 1

    mri[
        :,
        0,
        4:12,
        4:12,
        4:12,
    ] = 1.0

    dataset = TensorDataset(
        mri,
        target,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    epoch_loss = evaluate_one_epoch(
        model=model,
        dataloader=loader,
    )

    after = dict(model.named_parameters())

    assert isinstance(epoch_loss, float)
    assert math.isfinite(epoch_loss)
    assert epoch_loss >= 0.0
    assert model.training is False

    assert all(
        torch.equal(
            before[name],
            after[name].detach(),
        )
        for name in before
    )
