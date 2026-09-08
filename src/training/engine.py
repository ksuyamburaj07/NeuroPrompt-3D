import torch
from torch import nn

from src.training.losses import binary_segmentation_loss


def train_one_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    mri: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Run one optimization step for a single training batch."""

    model.train()

    optimizer.zero_grad()

    logits = model(mri)

    loss = binary_segmentation_loss(
        logits,
        target,
    )

    loss.backward()

    optimizer.step()

    return loss.detach()

def evaluate_one_batch(
    model: nn.Module,
    mri: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one batch without updating model parameters."""

    model.eval()

    with torch.no_grad():
        logits = model(mri)

        loss = binary_segmentation_loss(
            logits,
            target,
        )

    return loss.detach()

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
) -> float:
    """Train the model for one complete pass through a DataLoader."""

    total_loss = 0.0
    batch_count = 0

    for mri, target in dataloader:
        loss = train_one_batch(
            model=model,
            optimizer=optimizer,
            mri=mri,
            target=target,
        )

        total_loss += loss.item()
        batch_count += 1

    if batch_count == 0:
        raise ValueError(
            "dataloader must contain at least one batch"
        )

    return total_loss / batch_count

def evaluate_one_epoch(
    model: nn.Module,
    dataloader,
) -> float:
    """Evaluate the model for one complete pass through a DataLoader."""

    total_loss = 0.0
    batch_count = 0

    for mri, target in dataloader:
        loss = evaluate_one_batch(
            model=model,
            mri=mri,
            target=target,
        )

        total_loss += loss.item()
        batch_count += 1

    if batch_count == 0:
        raise ValueError(
            "dataloader must contain at least one batch"
        )

    return total_loss / batch_count
