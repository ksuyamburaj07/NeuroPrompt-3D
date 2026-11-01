"""Research geometry: intersections of pial triangles with MRI slices.

Coordinates entering the intersection function must be canonical RAS+
voxel coordinates. This module does not authorize case access or
modify source surfaces. Provisional anatomy remains QC-only.
"""

import numpy as np


def world_to_voxel(
    vertices_world: np.ndarray,
    affine_ras: np.ndarray,
) -> np.ndarray:
    """Map physical RAS+ millimetres to canonical MRI voxel XYZ."""
    vertices = np.asarray(vertices_world, dtype=np.float64)
    affine = np.asarray(affine_ras, dtype=np.float64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Expected Nx3 world vertices.")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Expected a finite 4x4 affine.")
    if not np.isfinite(vertices).all():
        raise ValueError("Non-finite vertex coordinates.")

    inverse = np.linalg.inv(affine)
    return vertices @ inverse[:3, :3].T + inverse[:3, 3]


def intersect_slice(
    vertices_voxel: np.ndarray,
    faces: np.ndarray,
    axis: int,
    index: float,
    atol: float = 1e-6,
) -> np.ndarray:
    """Return Nx2x3 independent triangle-plane intersection segments.

    Coplanar triangles and isolated point contacts do not create
    segments. Result coordinates are canonical voxel XYZ, not pixels.
    """
    vertices = np.asarray(vertices_voxel, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Expected Nx3 vertices.")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Expected Mx3 triangular faces.")
    if axis not in (0, 1, 2):
        raise ValueError("Axis must be 0, 1 or 2.")
    if not np.isfinite(vertices).all() or not np.isfinite(index):
        raise ValueError("Coordinates must be finite.")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("Invalid intersection tolerance.")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("Face index outside vertex array.")

    # Select potentially intersected triangles before Python iteration.
    distances = vertices[:, axis] - index
    face_distances = distances[triangles]

    candidates = (
        (face_distances.min(axis=1) <= atol)
        & (face_distances.max(axis=1) >= -atol)
        & ~np.all(np.abs(face_distances) <= atol, axis=1)
    )

    segments = []

    for face, d in zip(
        triangles[candidates],
        face_distances[candidates],
    ):
        triangle = vertices[face]
        points = []

        for i, j in ((0, 1), (1, 2), (2, 0)):
            a = float(d[i])
            b = float(d[j])

            if abs(a) <= atol and abs(b) <= atol:
                points.extend((triangle[i], triangle[j]))
            elif abs(a) <= atol:
                points.append(triangle[i])
            elif abs(b) <= atol:
                points.append(triangle[j])
            elif (a < -atol and b > atol) or (
                a > atol and b < -atol
            ):
                fraction = a / (a - b)
                points.append(
                    triangle[i]
                    + fraction * (triangle[j] - triangle[i])
                )

        unique = []
        for point in points:
            if not any(
                np.linalg.norm(point - other) <= atol
                for other in unique
            ):
                unique.append(point)

        if len(unique) < 2:
            continue

        # Degenerate edge/vertex cases: retain the longest segment.
        pairs = [
            (unique[i], unique[j])
            for i in range(len(unique))
            for j in range(i + 1, len(unique))
        ]
        start, end = max(
            pairs,
            key=lambda pair: np.sum((pair[1] - pair[0]) ** 2),
        )

        if np.linalg.norm(end - start) > atol:
            segments.append((start, end))

    if not segments:
        return np.empty((0, 2, 3), dtype=np.float64)

    return np.asarray(segments, dtype=np.float64)


def project_to_image(
    segments_voxel: np.ndarray,
    plane: str,
    shape_ras_xyz: tuple[int, int, int],
) -> np.ndarray:
    """Map Nx2x3 voxel segments to displayed MRI pixel coordinates.

    Matches viewer_service._oriented_slice:
    horizontal axes increase rightward; vertical axes are flipped.
    """
    segments = np.asarray(segments_voxel, dtype=np.float64)

    if segments.ndim != 3 or segments.shape[1:] != (2, 3):
        raise ValueError("Expected Nx2x3 segments.")
    if len(shape_ras_xyz) != 3 or any(
        int(size) <= 0 for size in shape_ras_xyz
    ):
        raise ValueError("Invalid MRI shape.")

    size_x, size_y, size_z = map(int, shape_ras_xyz)

    if plane == "axial":
        horizontal = segments[:, :, 0]
        vertical = (size_y - 1) - segments[:, :, 1]
    elif plane == "coronal":
        horizontal = segments[:, :, 0]
        vertical = (size_z - 1) - segments[:, :, 2]
    elif plane == "sagittal":
        horizontal = segments[:, :, 1]
        vertical = (size_z - 1) - segments[:, :, 2]
    else:
        raise ValueError("Unsupported anatomical plane.")

    return np.stack((horizontal, vertical), axis=-1)
