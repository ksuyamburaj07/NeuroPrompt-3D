"""Tests for shared MRI contrast without source modifications."""

import numpy as np
import pytest

from app.backend.services.frozen_display_contrast import (
    display_grayscale,
    volume_contrast_window,
)


def test_volume_window_ignores_zero_background():
    volume = np.array(
        [0, 10, 20, 40, 100],
        dtype=np.float32,
    ).reshape(1, 1, 5)

    expected = np.percentile(
        [10, 20, 40, 100],
        (1, 99),
    )

    actual = volume_contrast_window(volume)

    assert np.allclose(actual, expected)


def test_same_voxel_has_same_grayscale_in_all_planes():
    volume = np.arange(
        6 * 8 * 7,
        dtype=np.float32,
    ).reshape(6, 8, 7)

    volume[0, :, :] = 0

    x, y, z = 3, 4, 5
    window = volume_contrast_window(volume)

    views = (
        (volume[:, :, z].T[::-1, :], (7 - y, x)),
        (volume[:, y, :].T[::-1, :], (6 - z, x)),
        (volume[x, :, :].T[::-1, :], (6 - z, y)),
    )

    values = []

    for section, position in views:
        gray = display_grayscale(section, window)
        values.append(int(gray[position]))

    assert values[0] == values[1] == values[2]


def test_all_zero_volume():
    volume = np.zeros((3, 4, 5), dtype=np.float32)

    window = volume_contrast_window(volume)

    assert window == (0.0, 0.0)

    gray = display_grayscale(volume[:, :, 2], window)

    assert np.count_nonzero(gray) == 0


def test_constant_nonzero_volume():
    volume = np.full(
        (3, 4, 5),
        7.0,
        dtype=np.float32,
    )

    window = volume_contrast_window(volume)

    assert window == (7.0, 7.0)

    gray = display_grayscale(volume[:, :, 2], window)

    assert np.count_nonzero(gray) == 0


def test_inputs_are_not_modified():
    volume = np.arange(
        60, dtype=np.float32
    ).reshape(3, 4, 5)

    original = volume.copy()
    section = volume[:, :, 2]
    section_original = section.copy()

    window = volume_contrast_window(volume)
    display_grayscale(section, window)

    assert np.array_equal(volume, original)
    assert np.array_equal(section, section_original)


@pytest.mark.parametrize(
    "volume",
    [
        np.ones((3, 4), dtype=np.float32),
        np.array([[[np.nan]]], dtype=np.float32),
        np.array([[[np.inf]]], dtype=np.float32),
    ],
)
def test_invalid_volumes_rejected(volume):
    with pytest.raises(ValueError):
        volume_contrast_window(volume)


@pytest.mark.parametrize(
    "section,window",
    [
        (np.ones((2, 3, 4)), (0.0, 1.0)),
        (np.array([[np.nan]]), (0.0, 1.0)),
        (np.array([[np.inf]]), (0.0, 1.0)),
        (np.ones((2, 3)), (0.0, float("nan"))),
        (np.ones((2, 3)), (100.0, 10.0)),
    ],
)
def test_invalid_slices_or_windows_rejected(
    section, window
):
    with pytest.raises(ValueError):
        display_grayscale(section, window)
