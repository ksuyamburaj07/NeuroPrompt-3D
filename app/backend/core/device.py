"""Execution-device selection without changing frozen scientific parameters."""

import os

import torch


def resolve_execution_device() -> torch.device:
    requested = (
        os.environ.get(
            "NEUROPROMPT_DEVICE",
            "auto",
        )
        .strip()
        .lower()
    )

    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "NEUROPROMPT_DEVICE=cuda was requested "
                "but CUDA is unavailable."
            )

        return torch.device("cuda")

    raise ValueError(
        "NEUROPROMPT_DEVICE must be "
        "'auto', 'cpu', or 'cuda'."
    )
