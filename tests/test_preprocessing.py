import pytest
import torch

from src.data.preprocessing import (
    normalize_foreground_zscore,
    normalize_multimodal_foreground_zscore,
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
