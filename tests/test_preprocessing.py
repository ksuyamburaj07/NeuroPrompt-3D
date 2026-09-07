import pytest
import torch

from src.data.preprocessing import (
    normalize_foreground_zscore,
    normalize_multimodal_foreground_zscore,
    whole_tumor_mask,
    prepare_model_case,
    center_crop_pair,
    crop_pair_around_center,
    tumor_centered_crop,
    background_centered_crop,
    sample_training_patch,
)


def test_normalize_foreground_zscore_normalizes_only_nonzero_voxels():
    volume = torch.tensor(
        [[[0.0, 1.0],
          [3.0, 0.0]]],
        dtype=torch.float32,
    )

    normalized = normalize_foreground_zscore(volume)

    expected = torch.tensor(
        [[[0.0, -1.0],
          [1.0, 0.0]]],
        dtype=torch.float32,
    )

    assert torch.allclose(normalized, expected)

def test_normalize_foreground_zscore_keeps_all_zero_volume_zero():
    volume = torch.zeros((2, 2, 2), dtype=torch.float32)

    normalized = normalize_foreground_zscore(volume)

    assert torch.equal(normalized, volume)

def test_normalize_foreground_zscore_handles_zero_std():
    volume = torch.tensor(
        [[[0.0, 5.0],
          [5.0, 0.0]]],
        dtype=torch.float32,
    )

    normalized = normalize_foreground_zscore(volume)

    expected = torch.zeros_like(volume)

    assert torch.equal(normalized, expected)

def test_normalize_foreground_zscore_has_zero_mean_and_unit_std():
    volume = torch.tensor(
        [[[0.0, 1.0, 2.0],
          [3.0, 4.0, 0.0]]],
        dtype=torch.float32,
    )

    normalized = normalize_foreground_zscore(volume)

    original_foreground = volume != 0
    normalized_foreground = normalized[original_foreground]

    assert normalized.shape == volume.shape
    assert normalized.dtype == volume.dtype
    assert torch.equal(normalized[~original_foreground], torch.zeros(2))
    assert torch.isclose(
        normalized_foreground.mean(),
        torch.tensor(0.0),
        atol=1e-6,
    )
    assert torch.isclose(
        normalized_foreground.std(correction=0),
        torch.tensor(1.0),
        atol=1e-6,
    )

def test_normalize_multimodal_foreground_zscore_normalizes_each_channel_independently():
    mri = torch.tensor(
        [
            [[[0.0, 1.0, 2.0],
              [3.0, 4.0, 0.0]]],

            [[[0.0, 10.0, 20.0],
              [30.0, 40.0, 0.0]]],

            [[[0.0, 100.0, 200.0],
              [300.0, 400.0, 0.0]]],

            [[[0.0, 1000.0, 2000.0],
              [3000.0, 4000.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    normalized = normalize_multimodal_foreground_zscore(mri)

    assert normalized.shape == mri.shape
    assert normalized.dtype == mri.dtype

    for channel in normalized:
        foreground = channel != 0
        values = channel[foreground]

        assert torch.isclose(
            values.mean(),
            torch.tensor(0.0),
            atol=1e-6,
        )
        assert torch.isclose(
            values.std(correction=0),
            torch.tensor(1.0),
            atol=1e-6,
        )

@pytest.mark.parametrize(
    "shape",
    [
        (3, 2, 2, 2),
        (2, 2, 2),
    ],
)
def test_normalize_multimodal_foreground_zscore_rejects_invalid_shape(shape):
    mri = torch.zeros(shape, dtype=torch.float32)

    with pytest.raises(ValueError, match="MRI must have shape"):
        normalize_multimodal_foreground_zscore(mri)

def test_normalize_multimodal_foreground_zscore_does_not_modify_input():
    mri = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 0.0]]],
            [[[0.0, 2.0], [4.0, 0.0]]],
            [[[0.0, 3.0], [6.0, 0.0]]],
            [[[0.0, 4.0], [8.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    original = mri.clone()

    normalize_multimodal_foreground_zscore(mri)

    assert torch.equal(mri, original)

def test_normalize_foreground_zscore_rejects_non_3d_volume():
    volume = torch.zeros((2, 2), dtype=torch.float32)

    with pytest.raises(ValueError, match="Volume must be 3D"):
        normalize_foreground_zscore(volume)


def test_normalize_foreground_zscore_rejects_non_floating_dtype():
    volume = torch.ones((2, 2, 2), dtype=torch.int64)

    with pytest.raises(ValueError, match="floating-point"):
        normalize_foreground_zscore(volume)

def test_whole_tumor_mask_converts_all_nonzero_labels_to_one():
    segmentation = torch.tensor(
        [
            [
                [0, 1, 2],
                [3, 0, 1],
            ],
            [
                [2, 3, 0],
                [0, 0, 2],
            ],
        ],
        dtype=torch.uint8,
    )

    mask = whole_tumor_mask(segmentation)

    expected = torch.tensor(
        [
            [
                [0, 1, 1],
                [1, 0, 1],
            ],
            [
                [1, 1, 0],
                [0, 0, 1],
            ],
        ],
        dtype=torch.uint8,
    )

    assert torch.equal(mask, expected)

@pytest.mark.parametrize(
    "shape",
    [
        (5, 6),
        (1, 5, 6, 7),
    ],
    ids=[
        "2d",
        "4d",
    ],
)
def test_whole_tumor_mask_rejects_non_3d(shape):
    segmentation = torch.zeros(
        shape,
        dtype=torch.uint8,
    )

    with pytest.raises(
        ValueError,
        match="Segmentation must be 3D",
    ):
        whole_tumor_mask(segmentation)

def test_prepare_model_case_normalizes_mri_and_builds_binary_target():
    base = torch.arange(
        1,
        9,
        dtype=torch.float32,
    ).reshape(2, 2, 2)

    mri = torch.stack(
        [
            base,
            base + 10,
            base + 20,
            base + 30,
        ],
        dim=0,
    )

    segmentation = torch.tensor(
        [
            [
                [0, 1],
                [2, 0],
            ],
            [
                [3, 0],
                [1, 2],
            ],
        ],
        dtype=torch.uint8,
    )

    normalized_mri, target = prepare_model_case(
        mri,
        segmentation,
    )

    assert normalized_mri.shape == (4, 2, 2, 2)
    assert normalized_mri.dtype == torch.float32

    assert target.shape == (2, 2, 2)
    assert target.dtype == torch.uint8

    assert torch.equal(
        target,
        (segmentation > 0).to(torch.uint8),
    )

    for channel in normalized_mri:
        assert torch.isclose(
            channel.mean(),
            torch.tensor(0.0),
            atol=1e-6,
        )

        assert torch.isclose(
            channel.std(correction=0),
            torch.tensor(1.0),
            atol=1e-6,
        )

def test_prepare_model_case_rejects_spatial_shape_mismatch():
    mri = torch.zeros(
        (4, 4, 5, 6),
        dtype=torch.float32,
    )

    segmentation = torch.zeros(
        (4, 5, 5),
        dtype=torch.uint8,
    )

    with pytest.raises(
        ValueError,
        match="MRI and segmentation spatial dimensions must match",
    ):
        prepare_model_case(
            mri,
            segmentation,
        )

def test_center_crop_pair_uses_identical_spatial_region():
    mri = torch.arange(
        4 * 6 * 8 * 10,
        dtype=torch.float32,
    ).reshape(4, 6, 8, 10)

    target = torch.arange(
        6 * 8 * 10,
        dtype=torch.uint8,
    ).reshape(6, 8, 10)

    cropped_mri, cropped_target = center_crop_pair(
        mri,
        target,
        spatial_size=(4, 4, 6),
    )

    assert cropped_mri.shape == (4, 4, 4, 6)
    assert cropped_target.shape == (4, 4, 6)

    assert torch.equal(
        cropped_mri,
        mri[:, 1:5, 2:6, 2:8],
    )

    assert torch.equal(
        cropped_target,
        target[1:5, 2:6, 2:8],
    )

def test_center_crop_pair_rejects_crop_larger_than_volume():
    mri = torch.zeros(
        (4, 6, 8, 10),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (6, 8, 10),
        dtype=torch.uint8,
    )

    with pytest.raises(
        ValueError,
        match="Crop size must not exceed spatial dimensions",
    ):
        center_crop_pair(
            mri,
            target,
            spatial_size=(8, 8, 10),
        )

def test_crop_pair_around_center_keeps_selected_voxel_inside_patch():
    mri = torch.zeros(
        (4, 8, 8, 8),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (8, 8, 8),
        dtype=torch.uint8,
    )

    target[5, 4, 3] = 1

    cropped_mri, cropped_target = crop_pair_around_center(
        mri,
        target,
        center=(5, 4, 3),
        spatial_size=(4, 4, 4),
    )

    assert cropped_mri.shape == (4, 4, 4, 4)
    assert cropped_target.shape == (4, 4, 4)

    assert cropped_target.sum().item() == 1

def test_crop_pair_around_center_shifts_crop_inside_volume_near_edge():
    mri = torch.zeros(
        (4, 8, 8, 8),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (8, 8, 8),
        dtype=torch.uint8,
    )

    target[1, 1, 1] = 1
    cropped_mri, cropped_target = crop_pair_around_center(
        mri,
        target,
        center=(1, 1, 1),
        spatial_size=(4, 4, 4),
    )

    assert cropped_mri.shape == (4, 4, 4, 4)
    assert cropped_target.shape == (4, 4, 4)

    assert cropped_target.sum().item() == 1

def test_crop_pair_around_center_rejects_crop_larger_than_volume():
    mri = torch.zeros(
        (4, 6, 8, 10),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (6, 8, 10),
        dtype=torch.uint8,
    )

    with pytest.raises(
        ValueError,
        match="Crop size must not exceed spatial dimensions",
    ):
        crop_pair_around_center(
            mri,
            target,
            center=(3, 4, 5),
            spatial_size=(8, 8, 10),
        )

def test_tumor_centered_crop_keeps_tumor_inside_patch():
    mri = torch.zeros(
        (4, 8, 8, 8),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (8, 8, 8),
        dtype=torch.uint8,
    )

    target[5, 4, 3] = 1

    cropped_mri, cropped_target = tumor_centered_crop(
        mri,
        target,
        spatial_size=(4, 4, 4),
    )

    assert cropped_mri.shape == (4, 4, 4, 4)
    assert cropped_target.shape == (4, 4, 4)

    assert cropped_target.sum().item() == 1

def test_tumor_centered_crop_rejects_target_without_tumor():
    mri = torch.zeros(
        (4, 8, 8, 8),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (8, 8, 8),
        dtype=torch.uint8,
    )

    with pytest.raises(
        ValueError,
        match="Target contains no tumor voxels",
    ):
        tumor_centered_crop(
            mri,
            target,
            spatial_size=(4, 4, 4),
        )

def test_background_centered_crop_uses_nonzero_brain_background():
    mri = torch.zeros(
        (4, 8, 8, 8),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (8, 8, 8),
        dtype=torch.uint8,
    )

    # Simulate a small MRI foreground region.
    mri[:, 3:7, 3:7, 3:7] = 1.0

    # Put tumor inside part of that foreground.
    target[5, 5, 5] = 1

    cropped_mri, cropped_target = background_centered_crop(
        mri,
        target,
        spatial_size=(4, 4, 4),
    )

    assert cropped_mri.shape == (4, 4, 4, 4)
    assert cropped_target.shape == (4, 4, 4)

    assert cropped_mri.abs().sum().item() > 0

def test_background_centered_crop_rejects_no_valid_background():
    mri = torch.zeros(
        (4, 8, 8, 8),
        dtype=torch.float32,
    )

    target = torch.ones(
        (8, 8, 8),
        dtype=torch.uint8,
    )

    with pytest.raises(
        ValueError,
        match="No valid background voxels",
    ):
        background_centered_crop(
            mri,
            target,
            spatial_size=(4, 4, 4),
        )

def test_sample_training_patch_can_force_positive_or_background_branch():
    mri = torch.ones(
        (4, 12, 12, 12),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (12, 12, 12),
        dtype=torch.uint8,
    )

    # Put the tumor far from the first background region.
    target[10, 10, 10] = 1

    positive_mri, positive_target = sample_training_patch(
        mri,
        target,
        spatial_size=(4, 4, 4),
        positive_probability=1.0,
    )

    background_mri, background_target = sample_training_patch(
        mri,
        target,
        spatial_size=(4, 4, 4),
        positive_probability=0.0,
    )

    assert positive_mri.shape == (4, 4, 4, 4)
    assert positive_target.shape == (4, 4, 4)

    assert background_mri.shape == (4, 4, 4, 4)
    assert background_target.shape == (4, 4, 4)

    assert positive_target.sum().item() > 0
    assert background_target.sum().item() == 0

def test_tumor_centered_crop_is_reproducible_with_seeded_generator():
    mri = torch.arange(
        4 * 12 * 12 * 12,
        dtype=torch.float32,
    ).reshape(4, 12, 12, 12)

    target = torch.zeros(
        (12, 12, 12),
        dtype=torch.uint8,
    )

    target[2, 2, 2] = 1
    target[6, 6, 6] = 1
    target[9, 9, 9] = 1

    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)

    first_mri, first_target = tumor_centered_crop(
        mri,
        target,
        spatial_size=(4, 4, 4),
        generator=first_generator,
    )

    second_mri, second_target = tumor_centered_crop(
        mri,
        target,
        spatial_size=(4, 4, 4),
        generator=second_generator,
    )

    assert torch.equal(first_mri, second_mri)
    assert torch.equal(first_target, second_target)

def test_background_centered_crop_is_reproducible_with_seeded_generator():
    mri = torch.arange(
        4 * 12 * 12 * 12,
        dtype=torch.float32,
    ).reshape(4, 12, 12, 12)

    target = torch.zeros(
        (12, 12, 12),
        dtype=torch.uint8,
    )

    target[6, 6, 6] = 1

    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)

    first_mri, first_target = background_centered_crop(
        mri,
        target,
        spatial_size=(4, 4, 4),
        generator=first_generator,
    )

    second_mri, second_target = background_centered_crop(
        mri,
        target,
        spatial_size=(4, 4, 4),
        generator=second_generator,
    )

    assert torch.equal(first_mri, second_mri)
    assert torch.equal(first_target, second_target)

def test_sample_training_patch_is_reproducible_with_seeded_generator():
    mri = torch.arange(
        4 * 12 * 12 * 12,
        dtype=torch.float32,
    ).reshape(4, 12, 12, 12)

    target = torch.zeros(
        (12, 12, 12),
        dtype=torch.uint8,
    )

    target[2, 2, 2] = 1
    target[6, 6, 6] = 1
    target[9, 9, 9] = 1

    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)

    first_mri, first_target = sample_training_patch(
        mri,
        target,
        spatial_size=(4, 4, 4),
        positive_probability=0.5,
        generator=first_generator,
    )

    second_mri, second_target = sample_training_patch(
        mri,
        target,
        spatial_size=(4, 4, 4),
        positive_probability=0.5,
        generator=second_generator,
    )

    assert torch.equal(first_mri, second_mri)
    assert torch.equal(first_target, second_target)

@pytest.mark.parametrize(
    "positive_probability",
    [
        -0.1,
        1.1,
    ],
)
def test_sample_training_patch_rejects_invalid_positive_probability(
    positive_probability,
):
    mri = torch.ones(
        (4, 8, 8, 8),
        dtype=torch.float32,
    )

    target = torch.zeros(
        (8, 8, 8),
        dtype=torch.uint8,
    )

    target[4, 4, 4] = 1

    with pytest.raises(
        ValueError,
        match="positive_probability must be between 0 and 1",
    ):
        sample_training_patch(
            mri,
            target,
            spatial_size=(4, 4, 4),
            positive_probability=positive_probability,
        )
