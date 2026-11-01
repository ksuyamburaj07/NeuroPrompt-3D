"""Contrast-cache identity and source-authentication tests."""

from hashlib import sha256

import nibabel as nib
import numpy as np
import pytest

from app.backend.services.frozen_display_contrast import (
    _cached_volume_window,
    verified_volume_contrast_window,
)


@pytest.fixture(autouse=True)
def clear_contrast_cache():
    _cached_volume_window.cache_clear()
    yield
    _cached_volume_window.cache_clear()


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "source.nii.gz"

    volume = (
        np.arange(60, dtype=np.float32) + 1
    ).reshape(3, 4, 5)

    nib.save(
        nib.Nifti1Image(volume, np.eye(4)),
        str(path),
    )

    return path


def current_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_sha(path, expected):
    data = path.read_bytes()

    if sha256(data).hexdigest() != expected:
        raise ValueError("MRI checksum mismatch.")

    return data


def test_cache_hit_still_reverifies_source(source_file):
    expected = current_sha(source_file)
    verification_count = 0

    def verifier(path, digest):
        nonlocal verification_count
        verification_count += 1
        return verify_sha(path, digest)

    first = verified_volume_contrast_window(
        source_file,
        expected,
        verify_file=verifier,
    )

    second = verified_volume_contrast_window(
        source_file,
        expected,
        verify_file=verifier,
    )

    assert first == second
    assert verification_count == 2

    stats = _cached_volume_window.cache_info()
    assert stats.misses == 1
    assert stats.hits == 1

    assert stats.maxsize == 8


def test_tampering_rejected_even_after_cache_hit(source_file):
    original_sha = current_sha(source_file)

    verified_volume_contrast_window(
        source_file,
        original_sha,
        verify_file=verify_sha,
    )

    modified = np.full(
        (3, 4, 5),
        999.0,
        dtype=np.float32,
    )

    nib.save(
        nib.Nifti1Image(modified, np.eye(4)),
        str(source_file),
    )

    assert current_sha(source_file) != original_sha

    with pytest.raises(
        ValueError,
        match="MRI checksum mismatch",
    ):
        verified_volume_contrast_window(
            source_file,
            original_sha,
            verify_file=verify_sha,
        )

    # A stale cached window must not be returned.
    assert _cached_volume_window.cache_info().hits == 0


def test_new_digest_recalculates_window(source_file):
    original_sha = current_sha(source_file)

    first = verified_volume_contrast_window(
        source_file,
        original_sha,
        verify_file=verify_sha,
    )

    updated = (
        np.arange(60, dtype=np.float32) + 101
    ).reshape(3, 4, 5)

    nib.save(
        nib.Nifti1Image(updated, np.eye(4)),
        str(source_file),
    )

    new_sha = current_sha(source_file)
    assert new_sha != original_sha

    second = verified_volume_contrast_window(
        source_file,
        new_sha,
        verify_file=verify_sha,
    )

    assert second != first
    assert second[0] > first[0]
    assert second[1] > first[1]

    assert _cached_volume_window.cache_info().misses == 2


def test_invalid_digest_rejected_before_verification(source_file):
    def forbidden(*args):
        raise AssertionError(
            "Invalid digest reached the file verifier."
        )

    with pytest.raises(ValueError):
        verified_volume_contrast_window(
            source_file,
            "invalid",
            verify_file=forbidden,
        )
