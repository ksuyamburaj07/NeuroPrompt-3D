"""Frozen baseline model loading and deterministic MRI-only inference.

This module deliberately reuses the already-validated NeuroPrompt-3D
baseline implementation:

- src.data.nifti.load_multimodal_case
- src.data.preprocessing.normalize_multimodal_foreground_zscore
- src.training.checkpoint.read_training_checkpoint
- src.training.setup.build_baseline_model
- src.inference.sliding_window.sliding_window_logits

No target/segmentation loader is imported.

The frozen best baseline checkpoint is identified by SHA-256 before
deserialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from src.data.nifti import (
    load_multimodal_case,
)

from src.data.preprocessing import (
    normalize_multimodal_foreground_zscore,
)

from src.inference.sliding_window import (
    sliding_window_logits,
)

from src.training.checkpoint import (
    read_training_checkpoint,
)

from src.training.config import (
    BaselineTrainingConfig,
)

from src.training.setup import (
    build_baseline_model,
)


FROZEN_BEST_BASELINE_SHA256 = (
    "3834e3215d5d6610e8d952df2e61d7e7"
    "88de6caaa8db8d9e418e9a3240f60771"
)

FROZEN_BASELINE_THRESHOLD = 0.5


class BaselineIntegrityError(ValueError):
    """Hard failure for checkpoint, model, or inference integrity."""


@dataclass(frozen=True)
class BaselineCheckpointBundle:
    """Strictly loaded baseline model plus provenance."""

    model: torch.nn.Module
    config: BaselineTrainingConfig
    checkpoint_sha256: str
    epoch: int
    validation_loss: float


@dataclass(frozen=True)
class DeterministicBaselineResult:
    """Frozen deterministic baseline prediction."""

    probability_zyx: np.ndarray
    coarse_mask_zyx: np.ndarray


def sha256_file(
    path: str | Path,
) -> str:
    """Return SHA-256 for one file."""

    file_path = Path(
        path
    )

    if not file_path.is_file():
        raise BaselineIntegrityError(
            f"File does not exist: {file_path}"
        )

    digest = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as stream:
        while True:
            block = stream.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_baseline_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = FROZEN_BEST_BASELINE_SHA256,
) -> BaselineCheckpointBundle:
    """Verify hash, construct canonical model, and load state strictly."""

    path = Path(
        checkpoint_path
    )

    expected = str(
        expected_sha256
    ).lower()

    if (
        len(expected) != 64
        or any(
            character
            not in "0123456789abcdef"
            for character in expected
        )
    ):
        raise BaselineIntegrityError(
            "expected_sha256 must be a lowercase/hex-compatible "
            "64-character SHA-256 digest."
        )

    observed = sha256_file(
        path
    )

    if observed != expected:
        raise BaselineIntegrityError(
            "Baseline checkpoint SHA-256 mismatch."
        )

    checkpoint = (
        read_training_checkpoint(
            path,
            map_location="cpu",
        )
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise BaselineIntegrityError(
            "Baseline checkpoint must deserialize to a dictionary."
        )

    required_keys = {
        "epoch",
        "validation_loss",
        "config",
        "model_state_dict",
    }

    missing_keys = sorted(
        required_keys
        - set(
            checkpoint
        )
    )

    if missing_keys:
        raise BaselineIntegrityError(
            "Baseline checkpoint is missing required keys: "
            f"{missing_keys}"
        )

    config_payload = checkpoint[
        "config"
    ]

    if not isinstance(
        config_payload,
        dict,
    ):
        raise BaselineIntegrityError(
            "Baseline checkpoint config must be a dictionary."
        )

    try:
        config = BaselineTrainingConfig(
            **config_payload
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise BaselineIntegrityError(
            "Baseline checkpoint config is malformed."
        ) from exc

    model = build_baseline_model(
        config
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    if not isinstance(
        state_dict,
        dict,
    ):
        raise BaselineIntegrityError(
            "model_state_dict must be a dictionary."
        )

    try:
        load_result = (
            model.load_state_dict(
                state_dict,
                strict=True,
            )
        )

    except RuntimeError as exc:
        raise BaselineIntegrityError(
            "Baseline model state failed strict loading."
        ) from exc

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise BaselineIntegrityError(
            "Strict baseline state loading reported incompatible keys."
        )

    epoch = int(
        checkpoint[
            "epoch"
        ]
    )

    validation_loss = float(
        checkpoint[
            "validation_loss"
        ]
    )

    if epoch < 0:
        raise BaselineIntegrityError(
            "Checkpoint epoch must be non-negative."
        )

    if not math.isfinite(
        validation_loss
    ):
        raise BaselineIntegrityError(
            "Checkpoint validation_loss must be finite."
        )

    model.eval()

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    return BaselineCheckpointBundle(
        model=model,
        config=config,
        checkpoint_sha256=observed,
        epoch=epoch,
        validation_loss=validation_loss,
    )


def load_prepared_multimodal_mri(
    paths_by_modality: Mapping[
        str,
        str | Path | None,
    ],
) -> torch.Tensor:
    """Load and normalize only the four frozen baseline MRI modalities.

    Returns:
        contiguous float32 tensor [4,D,H,W].
    """

    mri = load_multimodal_case(
        paths_by_modality
    )

    if (
        mri.ndim != 4
        or int(
            mri.shape[0]
        ) != 4
    ):
        raise BaselineIntegrityError(
            "MRI loader returned unexpected tensor geometry."
        )

    if mri.dtype != torch.float32:
        raise BaselineIntegrityError(
            "MRI loader must return float32."
        )

    if not torch.isfinite(
        mri
    ).all():
        raise BaselineIntegrityError(
            "Loaded MRI contains non-finite values."
        )

    normalized = (
        normalize_multimodal_foreground_zscore(
            mri
        )
    )

    if (
        normalized.shape
        != mri.shape
    ):
        raise BaselineIntegrityError(
            "MRI normalization changed tensor geometry."
        )

    if not torch.isfinite(
        normalized
    ).all():
        raise BaselineIntegrityError(
            "Normalized MRI contains non-finite values."
        )

    return (
        normalized
        .to(
            dtype=torch.float32
        )
        .contiguous()
    )


def run_deterministic_baseline(
    model: torch.nn.Module,
    config: BaselineTrainingConfig,
    prepared_mri_czyx: torch.Tensor,
    *,
    device: str | torch.device,
) -> DeterministicBaselineResult:
    """Run frozen deterministic baseline inference.

    The existing baseline sliding-window helper owns model.eval().
    Thresholding is sigmoid(logits) >= 0.5.
    """

    mri = prepared_mri_czyx

    if (
        mri.ndim != 4
        or int(
            mri.shape[0]
        ) != 4
    ):
        raise BaselineIntegrityError(
            "prepared_mri_czyx must have shape [4,D,H,W]."
        )

    if not torch.isfinite(
        mri
    ).all():
        raise BaselineIntegrityError(
            "prepared_mri_czyx contains non-finite values."
        )

    target_device = torch.device(
        device
    )

    model = model.to(
        target_device
    )

    batch = (
        mri
        .unsqueeze(
            0
        )
        .to(
            target_device
        )
    )

    logits = sliding_window_logits(
        model=model,
        mri=batch,
        roi_size=tuple(
            int(value)
            for value in config.patch_size
        ),
        overlap=float(
            config.validation_overlap
        ),
    )

    expected_shape = (
        1,
        1,
        *tuple(
            int(value)
            for value in mri.shape[1:]
        ),
    )

    if tuple(
        logits.shape
    ) != expected_shape:
        raise BaselineIntegrityError(
            "Deterministic baseline logits have unexpected geometry."
        )

    if not torch.isfinite(
        logits
    ).all():
        raise BaselineIntegrityError(
            "Deterministic baseline logits contain non-finite values."
        )

    probability = (
        torch.sigmoid(
            logits
        )[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    probability = np.ascontiguousarray(
        probability
    )

    if not np.isfinite(
        probability
    ).all():
        raise BaselineIntegrityError(
            "Deterministic baseline probability contains "
            "non-finite values."
        )

    coarse_mask = (
        probability
        >= FROZEN_BASELINE_THRESHOLD
    )

    return DeterministicBaselineResult(
        probability_zyx=probability,
        coarse_mask_zyx=np.ascontiguousarray(
            coarse_mask,
            dtype=bool,
        ),
    )
