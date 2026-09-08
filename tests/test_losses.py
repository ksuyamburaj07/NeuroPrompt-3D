import torch

from src.training.losses import binary_segmentation_loss


def test_binary_segmentation_loss_accepts_dataset_target_format():
    logits = torch.randn(
        2,
        1,
        8,
        8,
        8,
    )

    target = torch.randint(
        low=0,
        high=2,
        size=(2, 8, 8, 8),
        dtype=torch.uint8,
    )

    loss = binary_segmentation_loss(
        logits,
        target,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

def test_binary_segmentation_loss_is_lower_for_better_prediction():
    target = torch.zeros(
        (1, 4, 4, 4),
        dtype=torch.uint8,
    )

    target[:, 1:3, 1:3, 1:3] = 1

    good_logits = torch.full(
        (1, 1, 4, 4, 4),
        fill_value=-6.0,
    )

    good_logits[
        :,
        :,
        1:3,
        1:3,
        1:3,
    ] = 6.0

    bad_logits = -good_logits

    good_loss = binary_segmentation_loss(
        good_logits,
        target,
    )

    bad_loss = binary_segmentation_loss(
        bad_logits,
        target,
    )

    assert good_loss < bad_loss
