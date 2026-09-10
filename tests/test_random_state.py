import torch
from torch.utils.data import Dataset

from src.data.loaders import build_training_dataloader
from src.training.random_state import (
    capture_training_random_state,
    restore_training_random_state,
)


class DummyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        raise RuntimeError(
            "Dummy samples are not needed for this RNG test."
        )


def test_capture_and_restore_training_random_state():
    loader = build_training_dataloader(
        base_dataset=DummyDataset(),
        spatial_size=(96, 96, 96),
        batch_size=1,
        seed=42,
    )

    torch.manual_seed(123)

    saved_state = capture_training_random_state(
        train_loader=loader,
    )

    expected_global = torch.rand(1)

    expected_shuffle = torch.rand(
        1,
        generator=loader.generator,
    )

    expected_patch = torch.rand(
        1,
        generator=loader.dataset.generator,
    )

    # Advance all three RNG streams.
    torch.rand(10)

    torch.rand(
        10,
        generator=loader.generator,
    )

    torch.rand(
        10,
        generator=loader.dataset.generator,
    )

    restore_training_random_state(
        state=saved_state,
        train_loader=loader,
    )

    actual_global = torch.rand(1)

    actual_shuffle = torch.rand(
        1,
        generator=loader.generator,
    )

    actual_patch = torch.rand(
        1,
        generator=loader.dataset.generator,
    )

    assert torch.equal(
        actual_global,
        expected_global,
    )

    assert torch.equal(
        actual_shuffle,
        expected_shuffle,
    )

    assert torch.equal(
        actual_patch,
        expected_patch,
    )

def test_training_random_state_supports_cpu_only_environment(
    monkeypatch,
):
    loader = build_training_dataloader(
        base_dataset=DummyDataset(),
        spatial_size=(96, 96, 96),
        batch_size=1,
        seed=42,
    )

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    saved_state = capture_training_random_state(
        train_loader=loader,
    )

    assert (
        saved_state["torch_cuda_rng_state_all"]
        is None
    )

    # Restoring a CPU-only snapshot must not require CUDA.
    restore_training_random_state(
        state=saved_state,
        train_loader=loader,
    )
