import pytest

from src.data.cases import ordered_modality_paths
from src.data.modalities import MODALITY_NAMES


def test_modality_paths_follow_shared_order(tmp_path):
    paths = {
        modality: tmp_path / f"case_001_{modality.lower()}.nii.gz"
        for modality in MODALITY_NAMES
    }
    shuffled_paths = {
        "FLAIR": paths["FLAIR"],
        "T2": paths["T2"],
        "T1": paths["T1"],
        "T1ce": paths["T1ce"],
    }

    result = ordered_modality_paths(shuffled_paths)

    assert result == tuple(paths[modality] for modality in MODALITY_NAMES)


@pytest.mark.parametrize("missing_modality", MODALITY_NAMES)
def test_modality_paths_reject_a_missing_modality(tmp_path, missing_modality):
    paths = {
        modality: tmp_path / f"case_001_{modality.lower()}.nii.gz"
        for modality in MODALITY_NAMES
        if modality != missing_modality
    }

    with pytest.raises(ValueError, match=missing_modality):
        ordered_modality_paths(paths)


@pytest.mark.parametrize("missing_path", [None, "", "   "])
def test_modality_paths_reject_an_empty_path(tmp_path, missing_path):
    paths = {
        modality: tmp_path / f"case_001_{modality.lower()}.nii.gz"
        for modality in MODALITY_NAMES
    }
    paths["FLAIR"] = missing_path

    with pytest.raises(ValueError, match="FLAIR"):
        ordered_modality_paths(paths)
