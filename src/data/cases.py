"""Definitions for real multimodal MRI cases."""

from collections.abc import Mapping
from pathlib import Path

from src.data.modalities import MODALITY_NAMES


def ordered_modality_paths(
    paths_by_modality: Mapping[str, str | Path | None],
) -> tuple[Path, ...]:
    """Validate and return one case's paths in the shared modality order."""
    missing_modalities = [
        modality
        for modality in MODALITY_NAMES
        if modality not in paths_by_modality
        or paths_by_modality[modality] is None
        or not str(paths_by_modality[modality]).strip()
    ]
    if missing_modalities:
        missing_names = ", ".join(missing_modalities)
        raise ValueError(f"Missing required MRI modalities: {missing_names}")

    return tuple(Path(paths_by_modality[modality]) for modality in MODALITY_NAMES)
