import torch


def resolve_device(
    requested_device: str = "auto",
) -> torch.device:
    """Resolve a requested training device."""

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available"
            )

        return torch.device("cuda")

    raise ValueError(
        "requested_device must be one of: auto, cpu, cuda"
    )

def move_batch_to_device(
    mri: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one MRI/target batch to the selected device."""

    moved_mri = mri.to(
        device=device,
    )

    moved_target = target.to(
        device=device,
    )

    return moved_mri, moved_target
