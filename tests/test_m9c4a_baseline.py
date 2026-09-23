from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import src.inference.baseline as baseline_module

from src.inference.baseline import (
    FROZEN_BASELINE_THRESHOLD,
    BaselineIntegrityError,
    load_baseline_checkpoint,
    load_prepared_multimodal_mri,
    run_deterministic_baseline,
    sha256_file,
)

from src.training.config import (
    BaselineTrainingConfig,
)

from src.training.setup import (
    build_baseline_model,
)


def _write_synthetic_checkpoint(
    tmp_path: Path,
    *,
    mutate_state: bool = False,
    omit_model_state: bool = False,
):
    config = BaselineTrainingConfig(
        patch_size=(
            8,
            8,
            8,
        ),
        batch_size=1,
        num_epochs=3,
        learning_rate=1e-3,
        base_channels=8,
        dropout_probability=0.2,
        positive_probability=0.5,
        validation_overlap=0.25,
        seed=42,
        device="cpu",
    )

    model = build_baseline_model(
        config
    )

    state_dict = {
        key: value.detach().clone()
        for key, value in (
            model.state_dict().items()
        )
    }

    if mutate_state:
        state_dict[
            "unexpected.synthetic_parameter"
        ] = torch.zeros(
            1
        )

    checkpoint = {
        "epoch": 3,
        "train_loss": 0.5,
        "validation_loss": 0.25,
        "best_validation_loss": None,
        "config": asdict(
            config
        ),
        "optimizer_state_dict": {},
        "training_random_state": {},
    }

    if not omit_model_state:
        checkpoint[
            "model_state_dict"
        ] = state_dict

    path = (
        tmp_path
        / "synthetic_baseline.pt"
    )

    torch.save(
        checkpoint,
        path,
    )

    return (
        path,
        config,
        model,
    )


def _write_synthetic_modalities(
    tmp_path: Path,
):
    shape_xyz = (
        4,
        5,
        6,
    )

    affine = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    paths = {}

    for index, modality in enumerate(
        (
            "T1",
            "T1ce",
            "T2",
            "FLAIR",
        ),
        start=1,
    ):
        data = np.zeros(
            shape_xyz,
            dtype=np.float32,
        )

        data[
            1,
            2,
            3,
        ] = float(
            index
        )

        data[
            2,
            2,
            3,
        ] = float(
            index
            + 2
        )

        path = (
            tmp_path
            / f"synthetic-{modality}.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(
                data,
                affine,
            ),
            str(
                path
            ),
        )

        paths[
            modality
        ] = path

    return paths


def test_frozen_baseline_threshold():
    assert (
        FROZEN_BASELINE_THRESHOLD
        == 0.5
    )


def test_synthetic_checkpoint_hash_and_strict_load(tmp_path):
    path, config, original_model = (
        _write_synthetic_checkpoint(
            tmp_path
        )
    )

    digest = sha256_file(
        path
    )

    bundle = load_baseline_checkpoint(
        path,
        expected_sha256=digest,
    )

    assert (
        bundle.checkpoint_sha256
        == digest
    )

    assert bundle.epoch == 3

    assert np.isclose(
        bundle.validation_loss,
        0.25,
    )

    assert (
        bundle.config
        == config
    )

    original_state = (
        original_model.state_dict()
    )

    loaded_state = (
        bundle.model.state_dict()
    )

    assert (
        original_state.keys()
        == loaded_state.keys()
    )

    for key in original_state:
        assert torch.equal(
            original_state[
                key
            ],
            loaded_state[
                key
            ],
        )

    assert bundle.model.training is False

    assert all(
        not parameter.requires_grad
        for parameter in (
            bundle.model.parameters()
        )
    )


def test_checkpoint_hash_mismatch_hard_fails_before_use(tmp_path):
    path, _, _ = (
        _write_synthetic_checkpoint(
            tmp_path
        )
    )

    with pytest.raises(
        BaselineIntegrityError,
        match="SHA-256 mismatch",
    ):
        load_baseline_checkpoint(
            path,
            expected_sha256=(
                "0"
                * 64
            ),
        )


def test_missing_checkpoint_key_is_integrity_failure(tmp_path):
    path, _, _ = (
        _write_synthetic_checkpoint(
            tmp_path,
            omit_model_state=True,
        )
    )

    digest = sha256_file(
        path
    )

    with pytest.raises(
        BaselineIntegrityError,
        match="missing required keys",
    ):
        load_baseline_checkpoint(
            path,
            expected_sha256=digest,
        )


def test_strict_state_mismatch_is_integrity_failure(tmp_path):
    path, _, _ = (
        _write_synthetic_checkpoint(
            tmp_path,
            mutate_state=True,
        )
    )

    digest = sha256_file(
        path
    )

    with pytest.raises(
        BaselineIntegrityError,
        match="strict loading",
    ):
        load_baseline_checkpoint(
            path,
            expected_sha256=digest,
        )


def test_mri_only_loader_returns_frozen_czyx_geometry(tmp_path):
    paths = _write_synthetic_modalities(
        tmp_path
    )

    mri = load_prepared_multimodal_mri(
        paths
    )

    assert tuple(
        mri.shape
    ) == (
        4,
        6,
        5,
        4,
    )

    assert mri.dtype == torch.float32
    assert mri.is_contiguous()

    for channel in mri:
        foreground = (
            channel
            != 0
        )

        values = (
            channel[
                foreground
            ]
            .to(
                torch.float64
            )
        )

        assert values.numel() == 2

        assert abs(
            float(
                values.mean()
            )
        ) < 1e-7

        assert abs(
            float(
                values.std(
                    correction=0
                )
            )
            - 1.0
        ) < 1e-7

        assert torch.all(
            channel[
                ~foreground
            ]
            == 0
        )


def test_mri_loader_does_not_require_target_file(tmp_path):
    paths = _write_synthetic_modalities(
        tmp_path
    )

    assert not any(
        "seg"
        in path.name.lower()
        for path in (
            tmp_path.iterdir()
        )
    )

    mri = load_prepared_multimodal_mri(
        paths
    )

    assert tuple(
        mri.shape
    ) == (
        4,
        6,
        5,
        4,
    )


def test_deterministic_threshold_is_greater_equal_half(monkeypatch):
    config = BaselineTrainingConfig(
        patch_size=(
            4,
            4,
            4,
        ),
        validation_overlap=0.25,
    )

    model = torch.nn.Identity()

    mri = torch.zeros(
        (
            4,
            2,
            2,
            2,
        ),
        dtype=torch.float32,
    )

    logits = torch.zeros(
        (
            1,
            1,
            2,
            2,
            2,
        ),
        dtype=torch.float32,
    )

    logits[
        0,
        0,
        0,
        0,
        0,
    ] = -1.0

    logits[
        0,
        0,
        0,
        0,
        1,
    ] = 1.0

    def fake_sliding_window_logits(
        model,
        mri,
        roi_size,
        overlap,
    ):
        del model
        del roi_size
        del overlap

        assert tuple(
            mri.shape
        ) == (
            1,
            4,
            2,
            2,
            2,
        )

        return logits.clone()

    monkeypatch.setattr(
        baseline_module,
        "sliding_window_logits",
        fake_sliding_window_logits,
    )

    result = run_deterministic_baseline(
        model,
        config,
        mri,
        device="cpu",
    )

    assert (
        result.coarse_mask_zyx[
            0,
            0,
            0,
        ]
        == False
    )

    assert (
        result.coarse_mask_zyx[
            0,
            0,
            1,
        ]
        == True
    )

    # Zero logit -> sigmoid(0) == 0.5 -> foreground.
    assert (
        result.coarse_mask_zyx[
            1,
            1,
            1,
        ]
        == True
    )

    assert np.isclose(
        result.probability_zyx[
            1,
            1,
            1,
        ],
        0.5,
    )


def test_nonfinite_deterministic_logits_hard_fail(monkeypatch):
    config = BaselineTrainingConfig(
        patch_size=(
            4,
            4,
            4,
        ),
        validation_overlap=0.25,
    )

    model = torch.nn.Identity()

    mri = torch.zeros(
        (
            4,
            2,
            2,
            2,
        ),
        dtype=torch.float32,
    )

    def fake_sliding_window_logits(
        model,
        mri,
        roi_size,
        overlap,
    ):
        del model
        del mri
        del roi_size
        del overlap

        result = torch.zeros(
            (
                1,
                1,
                2,
                2,
                2,
            ),
            dtype=torch.float32,
        )

        result[
            0,
            0,
            0,
            0,
            0,
        ] = torch.nan

        return result

    monkeypatch.setattr(
        baseline_module,
        "sliding_window_logits",
        fake_sliding_window_logits,
    )

    with pytest.raises(
        BaselineIntegrityError,
        match="non-finite",
    ):
        run_deterministic_baseline(
            model,
            config,
            mri,
            device="cpu",
        )
