from pathlib import Path
from dataclasses import asdict

from src.training.checkpoint import (
    read_training_checkpoint,
    restore_training_checkpoint_state,
)
from src.training.random_state import restore_training_random_state

def validate_resume_configuration(
    checkpoint_config: dict[str, object],
    current_config,
) -> None:
    """Ensure resumed training uses compatible experiment settings."""
    current_config_dict = asdict(current_config)

    allowed_changes = {
        "num_epochs",
        "device",
    }

    compared_keys = (
        set(checkpoint_config)
        | set(current_config_dict)
    ) - allowed_changes

    mismatches = []

    for key in sorted(compared_keys):
        checkpoint_value = checkpoint_config.get(
            key,
            "<missing>",
        )

        current_value = current_config_dict.get(
            key,
            "<missing>",
        )

        if checkpoint_value != current_value:
            mismatches.append(
                f"{key}: "
                f"checkpoint={checkpoint_value!r}, "
                f"current={current_value!r}"
            )

    if mismatches:
        raise ValueError(
            "Cannot resume training because configuration "
            "does not match the checkpoint: "
            + "; ".join(mismatches)
        )

def restore_baseline_training_state(
    experiment,
    checkpoint_path: str | Path,
) -> dict:
    """Validate and restore a baseline training checkpoint."""
    checkpoint = read_training_checkpoint(
        path=checkpoint_path,
        map_location="cpu",
    )

    validate_resume_configuration(
        checkpoint_config=checkpoint["config"],
        current_config=experiment.config,
    )

    restore_training_checkpoint_state(
        checkpoint=checkpoint,
        model=experiment.model,
        optimizer=experiment.optimizer,
    )

    restore_training_random_state(
        state=checkpoint["training_random_state"],
        train_loader=experiment.train_loader,
    )

    return checkpoint
