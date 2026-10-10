"""Read-only binary PLY reconstruction of frozen live segmentation masks."""

from __future__ import annotations

import struct

import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

from app.backend.services.artifact_service import (
    ArtifactRunNotComplete,
    resolve_live_artifact,
)
from app.backend.services.run_service import load_live_run
from app.backend.services.viewer_service import _modality_path


MASK_LAYERS = {
    "baseline": "baseline_mask_nifti",
    "final": "final_mask_nifti",
    "removed": "removal_mask_nifti",
}

MAX_SOURCE_VOXELS = 20_000_000
MAX_PADDED_VOXELS = 5_000_000
MAX_TRIANGLES = 1_000_000


class ViewerMeshCaseMismatch(RuntimeError):
    """The run belongs to another staged case."""


class ViewerMeshGeometryError(RuntimeError):
    """A segmentation cannot be rendered safely as a mesh."""


def _load_aligned_binary_mask(case_id: str, artifact_path):
    source = nib.as_closest_canonical(
        nib.load(str(_modality_path(case_id, "t1n")))
    )
    mask_image = nib.as_closest_canonical(
        nib.load(str(artifact_path))
    )

    if len(source.shape) != 3 or len(mask_image.shape) != 3:
        raise ViewerMeshGeometryError(
            "3D MRI and segmentation geometry are required."
        )

    if (
        source.shape != mask_image.shape
        or np.prod(source.shape) > MAX_SOURCE_VOXELS
    ):
        raise ViewerMeshGeometryError(
            "Segmentation shape is incompatible or exceeds safety limits."
        )

    for affine in (source.affine, mask_image.affine):
        if (
            not np.isfinite(affine).all()
            or abs(np.linalg.det(affine[:3, :3])) < 1e-8
        ):
            raise ViewerMeshGeometryError("Invalid spatial affine.")

    if not np.allclose(
        source.affine, mask_image.affine,
        atol=1e-5, rtol=0,
    ):
        raise ViewerMeshGeometryError(
            "Segmentation and source MRI affines do not match."
        )

    values = np.asanyarray(mask_image.dataobj)

    if not np.isfinite(values).all():
        raise ViewerMeshGeometryError(
            "Segmentation contains nonfinite values."
        )

    if not np.all((values == 0) | (values == 1)):
        raise ViewerMeshGeometryError(
            "Expected a binary segmentation mask."
        )

    return mask_image.affine, values.astype(bool, copy=False)


def _binary_ply(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Serialize RAS-mm vertices and triangular faces as binary PLY."""

    vertices = np.asarray(vertices, dtype="<f4")
    faces = np.asarray(faces, dtype="<i4")

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ViewerMeshGeometryError("Invalid mesh vertex array.")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ViewerMeshGeometryError("Invalid mesh triangle array.")

    if not np.isfinite(vertices).all():
        raise ViewerMeshGeometryError("Nonfinite mesh vertices.")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment NeuroPrompt-3D scientific visualization\n"
        "comment vertex coordinates: physical RAS+ millimetres\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    # PLY triangle record: 1-byte vertex count, then three int32 indices.
    face_records = bytearray()
    for a, b, c in faces:
        face_records.extend(
            struct.pack("<Biii", 3, int(a), int(b), int(c))
        )

    return header + vertices.tobytes() + bytes(face_records)


def viewer_mesh_ply(
    case_id: str,
    run_id: str,
    layer: str,
) -> bytes:
    """Generate visualization geometry; never modify a scientific mask."""

    if layer not in MASK_LAYERS:
        raise ValueError("Unsupported segmentation mesh layer.")

    run = load_live_run(run_id)

    if run.case_id != case_id:
        raise ViewerMeshCaseMismatch(
            "Inference run belongs to a different MRI case."
        )

    if run.status != "complete":
        raise ArtifactRunNotComplete(
            "Segmentation meshes require a completed inference run."
        )

    artifact = resolve_live_artifact(
        run_id,
        MASK_LAYERS[layer],
    )

    affine, mask = _load_aligned_binary_mask(
        case_id, artifact.path
    )

    occupied = np.argwhere(mask)

    if len(occupied) == 0:
        raise ViewerMeshGeometryError(
            "The selected segmentation contains no surface."
        )

    lower = occupied.min(axis=0)
    upper = occupied.max(axis=0) + 1

    cropped = mask[
        lower[0]:upper[0],
        lower[1]:upper[1],
        lower[2]:upper[2],
    ]

    padded_shape = np.asarray(cropped.shape) + 2

    if np.prod(padded_shape) > MAX_PADDED_VOXELS:
        raise ViewerMeshGeometryError(
            "Segmentation exceeds the interactive mesh size limit."
        )

    padded = np.pad(cropped, 1, mode="constant")

    vertices, faces, _, _ = marching_cubes(
        padded.astype(np.float32),
        level=0.5,
        step_size=1,
        allow_degenerate=False,
        gradient_direction="descent",
    )

    if len(faces) > MAX_TRIANGLES:
        raise ViewerMeshGeometryError(
            "Extracted mesh exceeds the triangle safety limit."
        )

    native_ras_voxels = vertices + lower - 1

    world_vertices = nib.affines.apply_affine(
        affine, native_ras_voxels
    )

    return _binary_ply(world_vertices, faces)
