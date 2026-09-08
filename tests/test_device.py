import pytest
import torch

from src.training.device import (
    move_batch_to_device,
    resolve_device,
)


def test_resolve_device_accepts_explicit_cpu():
    device = resolve_device("cpu")

    assert device == torch.device("cpu")


def test_resolve_device_auto_uses_cpu_when_cuda_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    device = resolve_device("auto")

    assert device == torch.device("cpu")


def test_resolve_device_auto_uses_cuda_when_available(
    monkeypatch,
):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )

    device = resolve_device("auto")

    assert device == torch.device("cuda")


def test_resolve_device_rejects_unknown_device():
    with pytest.raises(ValueError):
        resolve_device("quantum_gpu")

def test_move_batch_to_device_moves_mri_and_target_without_changing_format():
    mri = torch.randn(
        2,
        4,
        8,
        8,
        8,
        dtype=torch.float32,
    )

    target = torch.randint(
        low=0,
        high=2,
        size=(2, 8, 8, 8),
        dtype=torch.uint8,
    )

    device = torch.device("cpu")

    moved_mri, moved_target = move_batch_to_device(
        mri=mri,
        target=target,
        device=device,
    )

    assert moved_mri.device == device
    assert moved_target.device == device

    assert moved_mri.shape == mri.shape
    assert moved_target.shape == target.shape

    assert moved_mri.dtype == torch.float32
    assert moved_target.dtype == torch.uint8
