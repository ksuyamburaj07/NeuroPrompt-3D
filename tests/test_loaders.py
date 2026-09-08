import torch
from torch.utils.data import TensorDataset

from src.data.loaders import (
    build_brats_training_dataloader,
    build_brats_validation_dataloader,
    build_training_dataloader,
)


def test_build_training_dataloader_returns_model_ready_patch_batches():
    mri = torch.ones(
        (4, 4, 12, 12, 12),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (4, 12, 12, 12),
        dtype=torch.uint8,
    )

    target[
        :,
        4:8,
        4:8,
        4:8,
    ] = 1

    base_dataset = TensorDataset(
        mri,
        target,
    )

    loader = build_training_dataloader(
        base_dataset=base_dataset,
        spatial_size=(8, 8, 8),
        batch_size=2,
        positive_probability=1.0,
        seed=42,
    )

    batch_mri, batch_target = next(
        iter(loader)
    )

    assert batch_mri.shape == (
        2,
        4,
        8,
        8,
        8,
    )

    assert batch_target.shape == (
        2,
        8,
        8,
        8,
    )

    assert batch_mri.dtype == torch.float32
    assert batch_target.dtype == torch.uint8

def test_training_dataloader_replays_same_batches_with_same_seed():
    mri = torch.zeros(
        (4, 4, 12, 12, 12),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (4, 12, 12, 12),
        dtype=torch.uint8,
    )

    for index in range(4):
        mri[
            index,
            :,
            1:11,
            1:11,
            1:11,
        ] = float(index + 1)

        target[
            index,
            2,
            2,
            2,
        ] = 1

        target[
            index,
            9,
            9,
            9,
        ] = 1

    base_dataset = TensorDataset(
        mri,
        target,
    )

    loader_a = build_training_dataloader(
        base_dataset=base_dataset,
        spatial_size=(4, 4, 4),
        batch_size=2,
        positive_probability=1.0,
        seed=42,
    )

    loader_b = build_training_dataloader(
        base_dataset=base_dataset,
        spatial_size=(4, 4, 4),
        batch_size=2,
        positive_probability=1.0,
        seed=42,
    )

    batches_a = list(loader_a)
    batches_b = list(loader_b)

    assert len(batches_a) == len(batches_b)

    for (
        batch_mri_a,
        batch_target_a,
    ), (
        batch_mri_b,
        batch_target_b,
    ) in zip(
        batches_a,
        batches_b,
    ):
        assert torch.equal(
            batch_mri_a,
            batch_mri_b,
        )

        assert torch.equal(
            batch_target_a,
            batch_target_b,
        )

def test_build_brats_training_dataloader_uses_frozen_train_split(
    tmp_path,
    monkeypatch,
):
    mri = torch.ones(
        (2, 4, 12, 12, 12),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (2, 12, 12, 12),
        dtype=torch.uint8,
    )

    target[
        :,
        4:8,
        4:8,
        4:8,
    ] = 1

    fake_base_dataset = TensorDataset(
        mri,
        target,
    )

    captured = {}

    class FakeBraTSDataset:
        @classmethod
        def from_manifest(
            cls,
            cases_root,
            manifest_path,
            split_name,
        ):
            captured["cases_root"] = cases_root
            captured["manifest_path"] = manifest_path
            captured["split_name"] = split_name

            return fake_base_dataset

    monkeypatch.setattr(
        "src.data.loaders.BraTSDataset",
        FakeBraTSDataset,
    )

    cases_root = tmp_path / "cases"
    manifest_path = tmp_path / "splits.json"

    loader = build_brats_training_dataloader(
        cases_root=cases_root,
        manifest_path=manifest_path,
        spatial_size=(8, 8, 8),
        batch_size=1,
        positive_probability=1.0,
        seed=42,
    )

    batch_mri, batch_target = next(
        iter(loader)
    )

    assert captured["cases_root"] == cases_root
    assert captured["manifest_path"] == manifest_path
    assert captured["split_name"] == "train"

    assert batch_mri.shape == (
        1,
        4,
        8,
        8,
        8,
    )

    assert batch_target.shape == (
        1,
        8,
        8,
        8,
    )

def test_build_brats_validation_dataloader_uses_frozen_validation_split(
    tmp_path,
    monkeypatch,
):
    mri = torch.stack(
        [
            torch.full(
                (4, 12, 12, 12),
                fill_value=1.0,
            ),
            torch.full(
                (4, 12, 12, 12),
                fill_value=2.0,
            ),
        ]
    )

    target = torch.zeros(
        (2, 12, 12, 12),
        dtype=torch.uint8,
    )

    fake_base_dataset = TensorDataset(
        mri,
        target,
    )

    captured = {}

    class FakeBraTSDataset:
        @classmethod
        def from_manifest(
            cls,
            cases_root,
            manifest_path,
            split_name,
        ):
            captured["cases_root"] = cases_root
            captured["manifest_path"] = manifest_path
            captured["split_name"] = split_name

            return fake_base_dataset

    monkeypatch.setattr(
        "src.data.loaders.BraTSDataset",
        FakeBraTSDataset,
    )

    cases_root = tmp_path / "cases"
    manifest_path = tmp_path / "splits.json"

    loader = build_brats_validation_dataloader(
        cases_root=cases_root,
        manifest_path=manifest_path,
    )

    batches = list(loader)

    assert captured["cases_root"] == cases_root
    assert captured["manifest_path"] == manifest_path
    assert captured["split_name"] == "validation"

    assert len(batches) == 2

    first_mri, first_target = batches[0]
    second_mri, second_target = batches[1]

    assert first_mri.shape == (
        1,
        4,
        12,
        12,
        12,
    )

    assert first_target.shape == (
        1,
        12,
        12,
        12,
    )

    assert torch.all(first_mri == 1.0)
    assert torch.all(second_mri == 2.0)
