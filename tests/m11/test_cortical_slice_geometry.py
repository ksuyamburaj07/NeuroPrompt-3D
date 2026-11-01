"""Synthetic regression checks for cortical triangle-plane geometry."""

import numpy as np
import pytest

from app.backend.services.cortical_slice_geometry import (
    intersect_slice,
    project_to_image,
    world_to_voxel,
)


def assert_endpoints(actual, expected):
    assert actual.shape == (1, 2, 3)

    observed = sorted(map(tuple, np.round(actual[0], 6)))
    wanted = sorted(map(tuple, np.round(expected, 6)))

    np.testing.assert_allclose(observed, wanted, atol=1e-6)


def test_regular_triangle_intersection():
    vertices = np.array([
        [0, 0, -1],
        [2, 0, 1],
        [0, 2, 1],
    ], dtype=float)

    result = intersect_slice(
        vertices,
        np.array([[0, 1, 2]]),
        axis=2,
        index=0,
    )

    assert_endpoints(result, [
        [1, 0, 0],
        [0, 1, 0],
    ])


def test_two_vertices_on_plane():
    vertices = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [0, 1, 1],
    ], dtype=float)

    result = intersect_slice(
        vertices,
        np.array([[0, 1, 2]]),
        axis=2,
        index=0,
    )

    assert_endpoints(result, [
        [0, 0, 0],
        [2, 0, 0],
    ])


def test_one_vertex_on_plane():
    vertices = np.array([
        [0, 0, 0],
        [2, 0, 1],
        [0, 2, -1],
    ], dtype=float)

    result = intersect_slice(
        vertices,
        np.array([[0, 1, 2]]),
        axis=2,
        index=0,
    )

    assert_endpoints(result, [
        [0, 0, 0],
        [1, 1, 0],
    ])


def test_non_intersecting_triangle():
    vertices = np.array([
        [0, 0, 1],
        [2, 0, 1],
        [0, 2, 1],
    ], dtype=float)

    result = intersect_slice(
        vertices,
        np.array([[0, 1, 2]]),
        axis=2,
        index=0,
    )

    assert result.shape == (0, 2, 3)


def test_coplanar_triangle_skipped():
    vertices = np.array([
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
    ], dtype=float)

    result = intersect_slice(
        vertices,
        np.array([[0, 1, 2]]),
        axis=2,
        index=0,
    )

    assert result.shape == (0, 2, 3)


def test_world_to_canonical_voxel():
    affine = np.array([
        [2, 0, 0, 10],
        [0, 3, 0, 20],
        [0, 0, 4, 30],
        [0, 0, 0, 1],
    ], dtype=float)

    actual = world_to_voxel(
        np.array([[12, 26, 42]], dtype=float),
        affine,
    )

    np.testing.assert_allclose(actual, [[1, 2, 3]])


@pytest.mark.parametrize(
    ("plane", "expected"),
    [
        ("axial", [[[5, 13], [1, 17]]]),
        ("coronal", [[[5, 22], [1, 26]]]),
        ("sagittal", [[[6, 22], [2, 26]]]),
    ],
)
def test_display_orientation(plane, expected):
    segments = np.array([
        [[5, 6, 7], [1, 2, 3]],
    ], dtype=float)

    actual = project_to_image(
        segments,
        plane,
        (10, 20, 30),
    )

    np.testing.assert_allclose(actual, expected)


def test_reject_bad_face_index():
    with pytest.raises(ValueError):
        intersect_slice(
            np.array([[0, 0, 0]], dtype=float),
            np.array([[0, 1, 2]]),
            axis=2,
            index=0,
        )
