import torch
import torch.nn.functional as F


def binary_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Combine binary cross-entropy and soft Dice loss."""

    if logits.ndim != 5:
        raise ValueError(
            "logits must have shape [B, 1, D, H, W]"
        )

    if logits.shape[1] != 1:
        raise ValueError(
            "logits must have exactly one output channel"
        )

    if target.ndim != 4:
        raise ValueError(
            "target must have shape [B, D, H, W]"
        )

    expected_target_shape = (
        logits.shape[0],
        logits.shape[2],
        logits.shape[3],
        logits.shape[4],
    )

    if tuple(target.shape) != expected_target_shape:
        raise ValueError(
            "target spatial shape must match logits"
        )

    target_float = target.unsqueeze(1).to(
        device=logits.device,
        dtype=logits.dtype,
    )

    bce_loss = F.binary_cross_entropy_with_logits(
        logits,
        target_float,
    )

    probabilities = torch.sigmoid(logits)

    spatial_dims = (2, 3, 4)

    intersection = (
        probabilities * target_float
    ).sum(dim=spatial_dims)

    predicted_sum = probabilities.sum(
        dim=spatial_dims
    )

    target_sum = target_float.sum(
        dim=spatial_dims
    )

    dice_score = (
        2.0 * intersection + smooth
    ) / (
        predicted_sum + target_sum + smooth
    )

    dice_loss = 1.0 - dice_score.mean()

    return bce_loss + dice_loss
