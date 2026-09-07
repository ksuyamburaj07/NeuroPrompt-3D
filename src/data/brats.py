"""BraTS-specific case and subject utilities."""

from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import random
import json

from src.data.modalities import MODALITY_NAMES
import re


_CASE_ID_PATTERN = re.compile(
    r"^BraTS-GLI-\d{5}-\d{3}$"
)

_SUBJECT_ID_PATTERN = re.compile(
    r"^BraTS-GLI-\d{5}$"
)

_BRATS_SUFFIX_BY_MODALITY = {
    "T1": "t1n",
    "T1ce": "t1c",
    "T2": "t2w",
    "FLAIR": "t2f",
}

_REQUIRED_SUFFIXES = (
    "t1n",
    "t1c",
    "t2w",
    "t2f",
    "seg",
)

def _validate_case_id(case_id: str) -> None:
    """Raise ValueError if a BraTS case ID is malformed."""
    if not _CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(
            f"Invalid BraTS case ID: {case_id}"
        )


def subject_id_from_case(case_id: str) -> str:
    """Return the subject ID by removing the final timepoint suffix."""
    _validate_case_id(case_id)
    return case_id.rsplit("-", 1)[0]


def timepoint_from_case(case_id: str) -> int:
    """Return the final BraTS timepoint suffix as an integer."""
    _validate_case_id(case_id)
    return int(case_id.rsplit("-", 1)[1])

def expected_case_filenames(case_id: str) -> dict[str, str]:
    """Return the expected BraTS filenames for one valid case ID."""
    _validate_case_id(case_id)

    return {
        suffix: f"{case_id}-{suffix}.nii.gz"
        for suffix in _REQUIRED_SUFFIXES
    }

def modality_paths_for_case(
    case_dir: str | Path,
    case_id: str,
) -> dict[str, Path]:
    """Return loader-ready modality paths for one BraTS case."""
    _validate_case_id(case_id)

    case_dir = Path(case_dir)

    return {
        modality: case_dir / f"{case_id}-{_BRATS_SUFFIX_BY_MODALITY[modality]}.nii.gz"
        for modality in MODALITY_NAMES
    }

def discover_case_ids_in_archive(
    archive_path: str | Path,
) -> tuple[str, ...]:
    """Return unique sorted BraTS case IDs found inside a ZIP archive."""
    archive_path = Path(archive_path)

    case_ids = set()

    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".nii.gz"):
                continue

            path = Path(name)
            case_id = path.parent.name

            if _CASE_ID_PATTERN.fullmatch(case_id):
                case_ids.add(case_id)

    return tuple(sorted(case_ids))

def validate_archive_cases(
    archive_path: str | Path,
) -> tuple[str, ...]:
    """Validate that every BraTS case contains all required files."""
    archive_path = Path(archive_path)

    files_by_case: dict[str, set[str]] = {}

    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".nii.gz"):
                continue

            path = Path(name)
            case_id = path.parent.name

            if not _CASE_ID_PATTERN.fullmatch(case_id):
                continue

            files_by_case.setdefault(case_id, set()).add(path.name)

    case_ids = tuple(sorted(files_by_case))

    for case_id in case_ids:
        expected = set(
            expected_case_filenames(case_id).values()
        )
        actual = files_by_case[case_id]

        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)

        if missing:
            raise ValueError(
                f"{case_id} missing required files: {missing}"
            )

        if unexpected:
            raise ValueError(
                f"{case_id} unexpected files: {unexpected}"
            )

    return case_ids

def group_cases_by_subject(
    case_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Group BraTS case IDs by subject and sort each subject's timepoints."""
    grouped: dict[str, list[str]] = defaultdict(list)

    for case_id in case_ids:
        subject_id = subject_id_from_case(case_id)
        grouped[subject_id].append(case_id)

    return {
        subject_id: tuple(
            sorted(
                cases,
                key=timepoint_from_case,
            )
        )
        for subject_id, cases in sorted(grouped.items())
    }

def split_subject_ids(
    subject_ids: tuple[str, ...],
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    seed: int = 42,
) -> dict[str, tuple[str, ...]]:
    """Split subject IDs reproducibly into train, validation, and test sets."""
    if (
        train_fraction <= 0
        or validation_fraction <= 0
        or train_fraction + validation_fraction >= 1
    ):
        raise ValueError(
            "Split fractions must be positive and sum to less than 1"
        )

    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError(
            "Subject IDs must not contain duplicate subjects"
        )

    for subject_id in subject_ids:
        if not _SUBJECT_ID_PATTERN.fullmatch(subject_id):
            raise ValueError(
                f"Invalid BraTS subject ID: {subject_id}"
            )

    shuffled = sorted(subject_ids)

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)

    train_end = int(total * train_fraction)
    validation_end = train_end + int(
        total * validation_fraction
    )

    return {
        "train": tuple(shuffled[:train_end]),
        "validation": tuple(
            shuffled[train_end:validation_end]
        ),
        "test": tuple(shuffled[validation_end:]),
    }

def build_split_manifest(
    subjects: dict[str, tuple[str, ...]],
    splits: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, object]]:
    """Build split metadata containing subject and case assignments."""
    expected_split_names = {
         "train",
         "validation",
         "test",
    }

    if set(splits) != expected_split_names:
        raise ValueError(
            "Split names must be exactly: "
            "train, validation, test"
        )

    all_subject_ids = (
        splits["train"]
        + splits["validation"]
        + splits["test"]
    )

    if len(all_subject_ids) != len(set(all_subject_ids)):
        raise ValueError(
            "Subject overlap detected across splits"
        )

    unknown_subject_ids = (
        set(all_subject_ids) - set(subjects)
    )

    if unknown_subject_ids:
        unknown_subject_id = sorted(
            unknown_subject_ids
        )[0]

        raise ValueError(
            f"Unknown subject ID: {unknown_subject_id}"
        )

    if set(all_subject_ids) != set(subjects):
        raise ValueError(
            "Subject coverage must match discovered subjects exactly"
        )

    manifest = {}

    for split_name in ("train", "validation", "test"):
        subject_ids = splits[split_name]

        for subject_id in subject_ids:
            if subject_id not in subjects:
                raise ValueError(
                    f"Unknown subject ID: {subject_id}"
                )

        case_ids = tuple(
            case_id
            for subject_id in subject_ids
            for case_id in subjects[subject_id]
        )

        manifest[split_name] = {
            "subject_ids": subject_ids,
            "case_ids": case_ids,
            "subject_count": len(subject_ids),
            "case_count": len(case_ids),
        }

    return manifest

def load_split_case_ids(
    manifest_path: str | Path,
    split_name: str,
) -> tuple[str, ...]:
    """Load saved case IDs for one split from a frozen manifest."""
    valid_split_names = {
        "train",
        "validation",
        "test",
    }

    if split_name not in valid_split_names:
        raise ValueError(
            "Split name must be one of: "
            "train, validation, test"
        )

    manifest_path = Path(manifest_path)

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    case_ids = manifest["splits"][split_name]["case_ids"]

    return tuple(case_ids)
