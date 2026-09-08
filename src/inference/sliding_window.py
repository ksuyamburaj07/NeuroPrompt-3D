import torch
from torch import nn
from monai.inferers import sliding_window_inference


def sliding_window_logits(
    model: nn.Module,
    mri: torch.Tensor,
    roi_size: tuple[int, int, int],
    overlap: float = 0.25,
) -> torch.Tensor:
    """Return full-volume logits using sliding-window inference."""

    model.eval()

    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=mri,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=overlap,
        )

    return logits
