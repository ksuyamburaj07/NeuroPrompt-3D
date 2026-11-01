"""Shared, display-only MRI grayscale normalization.

A single nonzero-voxel P1/P99 window is derived from a
3D MRI volume and reused across anatomical planes.

This module does not modify scientific source arrays.
"""

from functools import lru_cache
from pathlib import Path
from typing import Callable
import re

import nibabel as nib
import numpy as np


def volume_contrast_window(
    volume_xyz,
) -> tuple[float, float]:
    """Calculate a common P1/P99 window from a 3D MRI volume."""

    volume = np.asarray(volume_xyz, dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError("Expected a 3D MRI volume.")

    if not np.isfinite(volume).all():
        raise ValueError("MRI volume contains invalid intensities.")

    foreground = volume[volume != 0]

    if foreground.size == 0:
        return (0.0, 0.0)

    lo, hi = np.percentile(foreground, (1, 99))

    return float(lo), float(hi)


def display_grayscale(
    slice_2d,
    window: tuple[float, float],
) -> np.ndarray:
    """Render one 2D MRI slice with a shared contrast window."""

    image = np.asarray(slice_2d, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError("Expected a 2D MRI slice.")

    if not np.isfinite(image).all():
        raise ValueError("MRI slice contains invalid intensities.")

    if len(window) != 2:
        raise ValueError("Invalid contrast window.")

    lo, hi = (float(window[0]), float(window[1]))

    if (
        not np.isfinite(lo)
        or not np.isfinite(hi)
        or hi < lo
    ):
        raise ValueError("Invalid contrast window.")

    if hi == lo:
        return np.zeros_like(image, dtype=np.uint8)

    return (
        np.clip(
            (image - lo) / (hi - lo),
            0.0,
            1.0,
        ) * 255.0
    ).astype(np.uint8)


@lru_cache(maxsize=8)
def _cached_volume_window(
    source_path: str,
    expected_sha: str,
) -> tuple[float, float]:
    """Cache only the display window, keyed by file and digest.

    This private function is reached only after the public
    helper has verified the source bytes.
    """
    image = nib.load(source_path)

    return volume_contrast_window(image.dataobj)


def verified_volume_contrast_window(
    source_path: Path,
    expected_sha: str,
    *,
    verify_file: Callable[[Path, str], bytes],
) -> tuple[float, float]:
    """Authenticate the MRI before every contrast-cache lookup.

    Caller must independently enforce the frozen MRI geometry
    and case-identity contracts.

    Cache hits do not replace or bypass checksum verification.
    """
    if (
        not isinstance(source_path, Path)
        or not isinstance(expected_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
    ):
        raise ValueError("Invalid MRI contrast source identity.")

    # This runs on every call, including cache hits.
    verify_file(source_path, expected_sha)

    return _cached_volume_window(
        str(source_path.resolve(strict=True)),
        expected_sha,
    )
