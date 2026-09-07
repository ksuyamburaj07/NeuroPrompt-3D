import nibabel as nib
import numpy as np
import pytest
import torch

from src.data.modalities import MODALITY_NAMES
from src.data.nifti import load_multimodal_case, load_segmentation


@pytest.fixture
def modality_paths(tmp_path):
    affine = np.diag([1.25, 1.5, 2.0, 1.0])
    affine[:3, 3] = (200.0, -30.0, 40.0)
    paths = {}
    for index, modality in enumerate(MODALITY_NAMES):
        values = np.arange(120, dtype=np.int16).reshape(6, 5, 4)
        values += 1000 * (index + 1)
        path = tmp_path / f"case_001_{modality.lower()}.nii.gz"
        nib.save(nib.Nifti1Image(values, affine), path)
        paths[modality] = path
    return paths


def test_load_separate_modalities_preserves_values_and_spatial_axes(modality_paths):
    mri = load_multimodal_case(modality_paths)

    assert mri.shape == (4, 4, 5, 6)
    assert mri.dtype == torch.float32
    assert mri.device.type == "cpu"
    assert mri.is_contiguous()
    # File voxel [X=3, Y=2, Z=1] has offset 3*20 + 2*4 + 1 = 69.
    assert mri[:, 1, 2, 3].tolist() == [1069.0, 2069.0, 3069.0, 4069.0]
    assert mri[:, 3, 4, 5].tolist() == [1119.0, 2119.0, 3119.0, 4119.0]


def test_load_separate_modalities_uses_fixed_channel_order(modality_paths):
    reversed_paths = {
        modality: str(modality_paths[modality])
        for modality in reversed(MODALITY_NAMES)
    }

    mri = load_multimodal_case(reversed_paths)

    assert MODALITY_NAMES == ("T1", "T1ce", "T2", "FLAIR")
    assert mri[:, 0, 0, 0].tolist() == [1000.0, 2000.0, 3000.0, 4000.0]


@pytest.mark.parametrize("modality", MODALITY_NAMES)
@pytest.mark.parametrize("shape", [(6, 5), (6, 5, 4, 1)], ids=["2d", "4d_singleton"])
def test_load_separate_modalities_rejects_non_3d(modality_paths, modality, shape):
    path = modality_paths[modality]
    affine = nib.load(path).affine
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), path)

    with pytest.raises(ValueError, match=f"{modality} NIfTI must be 3D"):
        load_multimodal_case(modality_paths)


@pytest.mark.parametrize("modality", MODALITY_NAMES)
def test_load_separate_modalities_rejects_shape_mismatch(modality_paths, modality):
    path = modality_paths[modality]
    affine = nib.load(path).affine
    values = np.zeros((6, 5, 3), dtype=np.float32)
    nib.save(nib.Nifti1Image(values, affine), path)

    with pytest.raises(ValueError, match="spatial shape.*must match T1"):
        load_multimodal_case(modality_paths)


@pytest.mark.parametrize(
    "entry, change",
    [
        pytest.param((0, 3), 0.001, id="translation"),
        pytest.param((0, 0), 0.1, id="voxel_spacing"),
        pytest.param((0, 0), -2.5, id="axis_direction"),
    ],
)
def test_load_separate_modalities_rejects_affine_mismatch(
    modality_paths, entry, change
):
    path = modality_paths["FLAIR"]
    image = nib.load(path)
    affine = image.affine.copy()
    affine[entry] += change
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj).copy(), affine), path)

    with pytest.raises(ValueError, match="FLAIR NIfTI affine must match T1"):
        load_multimodal_case(modality_paths)


def test_load_separate_modalities_accepts_affine_rounding_noise(modality_paths):
    path = modality_paths["T2"]
    image = nib.load(path)
    affine = image.affine.copy()
    affine[0, 0] += 5e-6
    nib.save(nib.Nifti1Image(np.asarray(image.dataobj).copy(), affine), path)
    difference = abs(nib.load(path).affine[0, 0] - image.affine[0, 0])
    assert 0 < difference < 1e-5

    mri = load_multimodal_case(modality_paths)

    assert mri.shape == (4, 4, 5, 6)

def test_load_segmentation_preserves_labels_and_spatial_axes(tmp_path):
    values = np.zeros((6, 5, 4), dtype=np.uint8)

    values[3, 2, 1] = 1
    values[5, 4, 3] = 3

    affine = np.eye(4)
    path = tmp_path / "case_001_seg.nii.gz"

    nib.save(
        nib.Nifti1Image(values, affine),
        path,
    )

    segmentation = load_segmentation(path)

    assert segmentation.shape == (4, 5, 6)
    assert segmentation.dtype == torch.uint8
    assert segmentation.is_contiguous()

    assert segmentation[1, 2, 3].item() == 1
    assert segmentation[3, 4, 5].item() == 3

@pytest.mark.parametrize(
    "shape",
    [
        (6, 5),
        (6, 5, 4, 1),
    ],
    ids=[
        "2d",
        "4d_singleton",
    ],
)
def test_load_segmentation_rejects_non_3d(tmp_path, shape):
    values = np.zeros(
        shape,
        dtype=np.uint8,
    )

    path = tmp_path / "case_001_seg.nii.gz"

    nib.save(
        nib.Nifti1Image(
            values,
            np.eye(4),
        ),
        path,
    )

    with pytest.raises(
        ValueError,
        match="Segmentation NIfTI must be 3D",
    ):
        load_segmentation(path)

def test_load_segmentation_rejects_reference_affine_mismatch(tmp_path):
    reference_path = tmp_path / "t1.nii.gz"
    segmentation_path = tmp_path / "seg.nii.gz"

    reference_affine = np.eye(4)

    segmentation_affine = np.eye(4)
    segmentation_affine[0, 3] = 5.0

    nib.save(
        nib.Nifti1Image(
            np.zeros((6, 5, 4), dtype=np.float32),
            reference_affine,
        ),
        reference_path,
    )

    nib.save(
        nib.Nifti1Image(
            np.zeros((6, 5, 4), dtype=np.uint8),
            segmentation_affine,
        ),
        segmentation_path,
    )

    with pytest.raises(
        ValueError,
        match="Segmentation NIfTI affine must match T1",
    ):
        load_segmentation(
            segmentation_path,
            reference_path=reference_path,
        )

def test_load_segmentation_accepts_reference_affine_rounding_noise(tmp_path):
    reference_path = tmp_path / "t1.nii.gz"
    segmentation_path = tmp_path / "seg.nii.gz"

    reference_affine = np.eye(4)

    segmentation_affine = np.eye(4)
    segmentation_affine[0, 0] += 5e-6

    nib.save(
        nib.Nifti1Image(
            np.zeros((6, 5, 4), dtype=np.float32),
            reference_affine,
        ),
        reference_path,
    )

    nib.save(
        nib.Nifti1Image(
            np.zeros((6, 5, 4), dtype=np.uint8),
            segmentation_affine,
        ),
        segmentation_path,
    )

    segmentation = load_segmentation(
        segmentation_path,
        reference_path=reference_path,
    )

    assert segmentation.shape == (4, 5, 6)
