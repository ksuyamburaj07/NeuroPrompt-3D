import pytest
from src.data.brats import (
    expected_case_filenames,
    modality_paths_for_case,
    subject_id_from_case,
    timepoint_from_case,
    discover_case_ids_in_archive,
    validate_archive_cases,
    group_cases_by_subject,
    split_subject_ids,
    build_split_manifest,
)


def test_subject_id_from_case_removes_timepoint_suffix():
    subject_id = subject_id_from_case(
        "BraTS-GLI-00008-103"
    )

    assert subject_id == "BraTS-GLI-00008"

def test_timepoint_from_case_returns_final_suffix():
    timepoint = timepoint_from_case(
        "BraTS-GLI-00008-103"
    )

    assert timepoint == 103

@pytest.mark.parametrize(
    "case_id",
    [
        "hello-123",
        "BraTS-GLI-00008-abc",
        "BraTS-GLI-0008-103",
        "BraTS-GLI-00008",
    ],
)
def test_subject_id_from_case_rejects_invalid_case_id(case_id):
    with pytest.raises(ValueError, match="Invalid BraTS case ID"):
        subject_id_from_case(case_id)

@pytest.mark.parametrize(
    "case_id",
    [
        "hello-123",
        "BraTS-GLI-00008-abc",
        "BraTS-GLI-0008-103",
        "BraTS-GLI-00008",
    ],
)
def test_timepoint_from_case_rejects_invalid_case_id(case_id):
    with pytest.raises(ValueError, match="Invalid BraTS case ID"):
        timepoint_from_case(case_id)

def test_expected_case_filenames_returns_all_required_files():
    filenames = expected_case_filenames(
        "BraTS-GLI-03011-101"
    )

    assert filenames == {
        "t1n": "BraTS-GLI-03011-101-t1n.nii.gz",
        "t1c": "BraTS-GLI-03011-101-t1c.nii.gz",
        "t2w": "BraTS-GLI-03011-101-t2w.nii.gz",
        "t2f": "BraTS-GLI-03011-101-t2f.nii.gz",
        "seg": "BraTS-GLI-03011-101-seg.nii.gz",
    }

def test_expected_case_filenames_rejects_invalid_case_id():
    with pytest.raises(ValueError, match="Invalid BraTS case ID"):
        expected_case_filenames("BraTS-GLI-invalid")

def test_modality_paths_for_case_maps_brats_names_to_internal_modalities(tmp_path):
    case_id = "BraTS-GLI-03011-101"

    paths = modality_paths_for_case(
        tmp_path,
        case_id,
    )

    assert paths == {
        "T1": tmp_path / f"{case_id}-t1n.nii.gz",
        "T1ce": tmp_path / f"{case_id}-t1c.nii.gz",
        "T2": tmp_path / f"{case_id}-t2w.nii.gz",
        "FLAIR": tmp_path / f"{case_id}-t2f.nii.gz",
    }

def test_modality_paths_for_case_rejects_invalid_case_id(tmp_path):
    with pytest.raises(ValueError, match="Invalid BraTS case ID"):
        modality_paths_for_case(
            tmp_path,
            "BraTS-GLI-invalid",
        )

def test_discover_case_ids_in_archive_returns_unique_sorted_cases(tmp_path):
    from zipfile import ZipFile

    archive_path = tmp_path / "brats.zip"

    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "training/BraTS-GLI-00008-101/"
            "BraTS-GLI-00008-101-t1n.nii.gz",
            b"",
        )
        archive.writestr(
            "training/BraTS-GLI-00005-100/"
            "BraTS-GLI-00005-100-t1n.nii.gz",
            b"",
        )
        archive.writestr(
            "training/BraTS-GLI-00008-101/"
            "BraTS-GLI-00008-101-t1c.nii.gz",
            b"",
        )

    case_ids = discover_case_ids_in_archive(archive_path)

    assert case_ids == (
        "BraTS-GLI-00005-100",
        "BraTS-GLI-00008-101",
    )

def test_validate_archive_cases_rejects_missing_required_file(tmp_path):
    from zipfile import ZipFile

    case_id = "BraTS-GLI-00005-100"
    archive_path = tmp_path / "brats.zip"

    with ZipFile(archive_path, "w") as archive:
        for suffix in ("t1n", "t1c", "t2w", "t2f"):
            archive.writestr(
                f"training/{case_id}/{case_id}-{suffix}.nii.gz",
                b"",
            )

    with pytest.raises(
        ValueError,
        match="missing required files",
    ):
        validate_archive_cases(archive_path)

def test_validate_archive_cases_accepts_complete_cases(tmp_path):
    from zipfile import ZipFile

    case_ids = (
        "BraTS-GLI-00005-100",
        "BraTS-GLI-00008-101",
    )

    archive_path = tmp_path / "brats.zip"

    with ZipFile(archive_path, "w") as archive:
        for case_id in case_ids:
            for suffix in ("t1n", "t1c", "t2w", "t2f", "seg"):
                archive.writestr(
                    f"training/{case_id}/{case_id}-{suffix}.nii.gz",
                    b"",
                )

    validated = validate_archive_cases(archive_path)

    assert validated == case_ids

def test_validate_archive_cases_rejects_unexpected_nifti_file(tmp_path):
    from zipfile import ZipFile

    case_id = "BraTS-GLI-00005-100"
    archive_path = tmp_path / "brats.zip"

    with ZipFile(archive_path, "w") as archive:
        for suffix in ("t1n", "t1c", "t2w", "t2f", "seg"):
            archive.writestr(
                f"training/{case_id}/{case_id}-{suffix}.nii.gz",
                b"",
            )

        archive.writestr(
            f"training/{case_id}/{case_id}-extra.nii.gz",
            b"",
        )

    with pytest.raises(
        ValueError,
        match="unexpected files",
    ):
        validate_archive_cases(archive_path)

def test_group_cases_by_subject_keeps_longitudinal_cases_together():
    case_ids = (
        "BraTS-GLI-00008-103",
        "BraTS-GLI-00005-100",
        "BraTS-GLI-00008-100",
        "BraTS-GLI-00008-102",
        "BraTS-GLI-00008-101",
    )

    grouped = group_cases_by_subject(case_ids)

    assert grouped == {
        "BraTS-GLI-00005": (
            "BraTS-GLI-00005-100",
        ),
        "BraTS-GLI-00008": (
            "BraTS-GLI-00008-100",
            "BraTS-GLI-00008-101",
            "BraTS-GLI-00008-102",
            "BraTS-GLI-00008-103",
        ),
    }

def test_split_subject_ids_is_deterministic():
    subject_ids = tuple(
        f"BraTS-GLI-{index:05d}"
        for index in range(100)
    )

    first = split_subject_ids(
        subject_ids,
        train_fraction=0.80,
        validation_fraction=0.10,
        seed=42,
    )

    second = split_subject_ids(
        subject_ids,
        train_fraction=0.80,
        validation_fraction=0.10,
        seed=42,
    )

    assert first == second

    assert len(first["train"]) == 80
    assert len(first["validation"]) == 10
    assert len(first["test"]) == 10

def test_split_subject_ids_does_not_depend_on_input_order():
    subject_ids = tuple(
        f"BraTS-GLI-{index:05d}"
        for index in range(100)
    )

    reversed_subject_ids = tuple(reversed(subject_ids))

    first = split_subject_ids(
        subject_ids,
        train_fraction=0.80,
        validation_fraction=0.10,
        seed=42,
    )

    second = split_subject_ids(
        reversed_subject_ids,
        train_fraction=0.80,
        validation_fraction=0.10,
        seed=42,
    )

    assert first == second

def test_split_subject_ids_has_full_coverage_and_no_overlap():
    subject_ids = tuple(
        f"BraTS-GLI-{index:05d}"
        for index in range(100)
    )

    splits = split_subject_ids(
        subject_ids,
        train_fraction=0.80,
        validation_fraction=0.10,
        seed=42,
    )

    train = set(splits["train"])
    validation = set(splits["validation"])
    test = set(splits["test"])

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)

    combined = train | validation | test

    assert combined == set(subject_ids)

@pytest.mark.parametrize(
    "train_fraction, validation_fraction",
    [
        (0.0, 0.10),
        (-0.10, 0.10),
        (0.80, 0.0),
        (0.80, -0.10),
        (0.90, 0.20),
        (1.0, 0.10),
    ],
)
def test_split_subject_ids_rejects_invalid_fractions(
    train_fraction,
    validation_fraction,
):
    subject_ids = tuple(
        f"BraTS-GLI-{index:05d}"
        for index in range(100)
    )

    with pytest.raises(ValueError, match="Split fractions"):
        split_subject_ids(
            subject_ids,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            seed=42,
        )

def test_split_subject_ids_rejects_duplicate_subject_ids():
    subject_ids = (
        "BraTS-GLI-00008",
        "BraTS-GLI-00008",
        "BraTS-GLI-00009",
    )

    with pytest.raises(ValueError, match="duplicate subject"):
        split_subject_ids(
            subject_ids,
            train_fraction=0.80,
            validation_fraction=0.10,
            seed=42,
        )

@pytest.mark.parametrize(
    "subject_id",
    [
        "hello",
        "BraTS-GLI-0008",
        "BraTS-GLI-00008-100",
    ],
)
def test_split_subject_ids_rejects_invalid_subject_ids(subject_id):
    subject_ids = (
        "BraTS-GLI-00001",
        "BraTS-GLI-00002",
        subject_id,
    )

    with pytest.raises(ValueError, match="Invalid BraTS subject ID"):
        split_subject_ids(
            subject_ids,
            train_fraction=0.80,
            validation_fraction=0.10,
            seed=42,
        )

def test_build_split_manifest_maps_subjects_to_cases():
    subjects = {
        "BraTS-GLI-00001": (
            "BraTS-GLI-00001-100",
            "BraTS-GLI-00001-101",
        ),
        "BraTS-GLI-00002": (
            "BraTS-GLI-00002-100",
        ),
        "BraTS-GLI-00003": (
            "BraTS-GLI-00003-100",
        ),
    }

    splits = {
        "train": ("BraTS-GLI-00001",),
        "validation": ("BraTS-GLI-00002",),
        "test": ("BraTS-GLI-00003",),
    }

    manifest = build_split_manifest(
        subjects,
        splits,
    )

    assert manifest["train"]["subject_ids"] == (
        "BraTS-GLI-00001",
    )

    assert manifest["train"]["case_ids"] == (
        "BraTS-GLI-00001-100",
        "BraTS-GLI-00001-101",
    )

    assert manifest["train"]["subject_count"] == 1
    assert manifest["train"]["case_count"] == 2

    assert manifest["validation"]["case_count"] == 1
    assert manifest["test"]["case_count"] == 1

def test_build_split_manifest_rejects_unknown_subject():
    subjects = {
        "BraTS-GLI-00001": (
            "BraTS-GLI-00001-100",
        ),
    }

    splits = {
        "train": ("BraTS-GLI-00001",),
        "validation": ("BraTS-GLI-99999",),
        "test": (),
    }

    with pytest.raises(
        ValueError,
        match="Unknown subject ID",
    ):
        build_split_manifest(
            subjects,
            splits,
        )

@pytest.mark.parametrize(
    "splits",
    [
        {
            "train": ("BraTS-GLI-00001",),
            "validation": (),
        },
        {
            "train": ("BraTS-GLI-00001",),
            "validation": (),
            "test": (),
            "holdout": (),
        },
    ],
)
def test_build_split_manifest_rejects_invalid_split_names(splits):
    subjects = {
        "BraTS-GLI-00001": (
            "BraTS-GLI-00001-100",
        ),
    }

    with pytest.raises(ValueError, match="Split names"):
        build_split_manifest(
            subjects,
            splits,
        )

def test_build_split_manifest_rejects_subject_overlap():
    subjects = {
        "BraTS-GLI-00001": (
            "BraTS-GLI-00001-100",
        ),
        "BraTS-GLI-00002": (
            "BraTS-GLI-00002-100",
        ),
    }

    splits = {
        "train": ("BraTS-GLI-00001",),
        "validation": ("BraTS-GLI-00001",),
        "test": ("BraTS-GLI-00002",),
    }

    with pytest.raises(
        ValueError,
        match="Subject overlap",
    ):
        build_split_manifest(
            subjects,
            splits,
        )

def test_build_split_manifest_rejects_missing_subject():
    subjects = {
        "BraTS-GLI-00001": (
            "BraTS-GLI-00001-100",
        ),
        "BraTS-GLI-00002": (
            "BraTS-GLI-00002-100",
        ),
        "BraTS-GLI-00003": (
            "BraTS-GLI-00003-100",
        ),
    }

    splits = {
        "train": ("BraTS-GLI-00001",),
        "validation": ("BraTS-GLI-00002",),
        "test": (),
    }

    with pytest.raises(
        ValueError,
        match="Subject coverage",
    ):
        build_split_manifest(
            subjects,
            splits,
        )
