"""Application-facing modality names mapped to the frozen research core."""

from src.data.modalities import MODALITY_NAMES


APP_MODALITY_TO_CORE = {
    "t1n": "T1",
    "t1c": "T1ce",
    "t2w": "T2",
    "t2f": "FLAIR",
}

APP_MODALITY_LABELS = {
    "t1n": "T1n",
    "t1c": "T1c",
    "t2w": "T2w",
    "t2f": "T2-FLAIR",
}

REQUIRED_APP_MODALITIES = tuple(APP_MODALITY_TO_CORE)


if tuple(APP_MODALITY_TO_CORE.values()) != MODALITY_NAMES:
    raise RuntimeError(
        "M11 application modality mapping no longer matches "
        "the frozen research-core channel order"
    )
