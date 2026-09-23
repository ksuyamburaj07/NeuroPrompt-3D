"""Frozen selective MC-Dropout inference primitives.

Frozen behavior:

- T = 10 stochastic probability predictions;
- master seed = 20260913;
- deterministic case-specific seed derived from case ID;
- only torch.nn.Dropout3d modules enter training mode;
- every other module remains in evaluation mode;
- sliding-window inference must preserve the caller's model mode;
- predictive variance is population variance across probabilities
  using float64 accumulation and returned as float32.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from monai.inferers import sliding_window_inference


MC_PASSES = 10
MC_MASTER_SEED = 20260913
EXPECTED_DROPOUT3D_MODULES = 5


class UncertaintyIntegrityError(ValueError):
    """Hard failure for malformed or non-finite uncertainty execution."""


@dataclass(frozen=True)
class MCUncertaintyResult:
    """Outputs from one frozen T=10 case-level MC sequence."""

    case_seed: int
    activated_dropout_names: tuple[str, ...]
    pass_hashes: tuple[str, ...]
    probability_stack: np.ndarray
    mean_probability: np.ndarray
    predictive_variance: np.ndarray


def derive_case_seed(
    case_id: str,
    master_seed: int = MC_MASTER_SEED,
) -> int:
    """Derive the frozen deterministic per-case MC seed.

    Frozen M8B1 rule:

    first 4 bytes of SHA256(
        "NeuroPrompt3D|M8B|<master_seed>|<case_id>"
    ),
    interpreted big-endian and masked to 31 bits.
    """

    case_text = str(
        case_id
    )

    if not case_text:
        raise UncertaintyIntegrityError(
            "case_id must be non-empty."
        )

    master = int(
        master_seed
    )

    payload = (
        f"NeuroPrompt3D|M8B|"
        f"{master}|"
        f"{case_text}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    seed = (
        int.from_bytes(
            digest[:4],
            byteorder="big",
            signed=False,
        )
        & 0x7FFFFFFF
    )

    return int(
        seed
    )


def enable_mc_dropout(
    model: torch.nn.Module,
) -> tuple[str, ...]:
    """Enable only Dropout3d modules while all else remains in eval."""

    model.eval()

    activated = []

    for name, module in model.named_modules():
        if isinstance(
            module,
            torch.nn.Dropout3d,
        ):
            module.train()
            activated.append(
                name
            )

    unexpected_training = [
        name
        for name, module in model.named_modules()
        if (
            module.training
            and not isinstance(
                module,
                torch.nn.Dropout3d,
            )
        )
    ]

    if unexpected_training:
        raise UncertaintyIntegrityError(
            "Unexpected non-Dropout3d modules "
            "entered training mode: "
            f"{unexpected_training}"
        )

    return tuple(
        activated
    )


def disable_mc_dropout(
    model: torch.nn.Module,
) -> None:
    """Return the complete model to evaluation mode."""

    model.eval()


def sliding_window_logits_mode_preserving(
    model: torch.nn.Module,
    mri_batch: torch.Tensor,
    roi_size: tuple[int, int, int],
    overlap: float,
) -> torch.Tensor:
    """Run MONAI sliding-window inference without changing model mode."""

    if mri_batch.ndim != 5:
        raise UncertaintyIntegrityError(
            "mri_batch must have shape [B,C,D,H,W]."
        )

    roi = tuple(
        int(value)
        for value in roi_size
    )

    if (
        len(roi) != 3
        or any(
            value <= 0
            for value in roi
        )
    ):
        raise UncertaintyIntegrityError(
            "roi_size must contain three positive integers."
        )

    overlap_value = float(
        overlap
    )

    if (
        not math.isfinite(
            overlap_value
        )
        or overlap_value < 0.0
        or overlap_value >= 1.0
    ):
        raise UncertaintyIntegrityError(
            "overlap must be finite and in [0, 1)."
        )

    if not torch.isfinite(
        mri_batch
    ).all():
        raise UncertaintyIntegrityError(
            "mri_batch contains non-finite values."
        )

    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=mri_batch,
            roi_size=roi,
            sw_batch_size=1,
            predictor=model,
            overlap=overlap_value,
        )

    if not torch.isfinite(
        logits
    ).all():
        raise UncertaintyIntegrityError(
            "Sliding-window logits contain non-finite values."
        )

    return logits


def probability_stack_statistics(
    probability_stack: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen MC mean and population predictive variance."""

    stack = np.asarray(
        probability_stack,
        dtype=np.float32,
    )

    if (
        stack.ndim != 4
        or stack.shape[0] != MC_PASSES
    ):
        raise UncertaintyIntegrityError(
            "probability_stack must have shape [10,D,H,W]."
        )

    if not np.isfinite(
        stack
    ).all():
        raise UncertaintyIntegrityError(
            "probability_stack contains non-finite values."
        )

    mean_probability = (
        np.mean(
            stack,
            axis=0,
            dtype=np.float64,
        )
        .astype(
            np.float32
        )
    )

    predictive_variance = (
        np.var(
            stack,
            axis=0,
            dtype=np.float64,
        )
        .astype(
            np.float32
        )
    )

    return (
        mean_probability,
        predictive_variance,
    )


def run_mc_dropout_case(
    model: torch.nn.Module,
    mri_batch: torch.Tensor,
    roi_size: tuple[int, int, int],
    overlap: float,
    case_id: str,
) -> MCUncertaintyResult:
    """Run the frozen T=10 selective MC-Dropout sequence for one case."""

    if (
        mri_batch.ndim != 5
        or int(
            mri_batch.shape[0]
        ) != 1
    ):
        raise UncertaintyIntegrityError(
            "Frozen MC execution requires batch size 1."
        )

    case_seed = derive_case_seed(
        case_id
    )

    cpu_rng_state = (
        torch.get_rng_state()
        .clone()
    )

    cuda_rng_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    stack = None
    pass_hashes = []

    try:
        torch.manual_seed(
            case_seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                case_seed
            )

        activated = enable_mc_dropout(
            model
        )

        if len(
            activated
        ) != EXPECTED_DROPOUT3D_MODULES:
            raise UncertaintyIntegrityError(
                "Expected exactly "
                f"{EXPECTED_DROPOUT3D_MODULES} "
                "Dropout3d modules, found "
                f"{len(activated)}."
            )

        for pass_index in range(
            MC_PASSES
        ):
            logits = (
                sliding_window_logits_mode_preserving(
                    model=model,
                    mri_batch=mri_batch,
                    roi_size=roi_size,
                    overlap=overlap,
                )
            )

            if (
                logits.ndim != 5
                or int(
                    logits.shape[0]
                ) != 1
                or int(
                    logits.shape[1]
                ) != 1
            ):
                raise UncertaintyIntegrityError(
                    "Frozen baseline logits must have "
                    "shape [1,1,D,H,W]."
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

            probability = (
                np.ascontiguousarray(
                    probability
                )
            )

            if not np.isfinite(
                probability
            ).all():
                raise UncertaintyIntegrityError(
                    "MC probability contains non-finite values."
                )

            if stack is None:
                stack = np.empty(
                    (
                        MC_PASSES,
                        *probability.shape,
                    ),
                    dtype=np.float32,
                )

            elif (
                probability.shape
                != stack.shape[1:]
            ):
                raise UncertaintyIntegrityError(
                    "MC pass geometry changed within a case."
                )

            stack[
                pass_index
            ] = probability

            pass_hashes.append(
                hashlib.sha256(
                    probability.tobytes(
                        order="C"
                    )
                ).hexdigest()
            )

        if len(
            set(
                pass_hashes
            )
        ) != MC_PASSES:
            raise UncertaintyIntegrityError(
                "Expected 10 distinct stochastic "
                "MC probability volumes."
            )

        if stack is None:
            raise UncertaintyIntegrityError(
                "MC probability stack was not created."
            )

        mean_probability, predictive_variance = (
            probability_stack_statistics(
                stack
            )
        )

        return MCUncertaintyResult(
            case_seed=case_seed,
            activated_dropout_names=activated,
            pass_hashes=tuple(
                pass_hashes
            ),
            probability_stack=stack,
            mean_probability=mean_probability,
            predictive_variance=predictive_variance,
        )

    finally:
        disable_mc_dropout(
            model
        )

        torch.set_rng_state(
            cpu_rng_state
        )

        if (
            cuda_rng_states
            is not None
        ):
            torch.cuda.set_rng_state_all(
                cuda_rng_states
            )
