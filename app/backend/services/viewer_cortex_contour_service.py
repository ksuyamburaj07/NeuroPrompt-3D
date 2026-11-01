"""Read-only, identity-gated cortical contours for canonical RAS+ MRI.

Only eligible registered cortex assets may be used. This module
does not approve provisional reconstructions or modify research data.
"""

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from app.backend.services.cortical_slice_geometry import (
    intersect_slice,
    project_to_image,
    world_to_voxel,
)
from app.backend.services.viewer_cortex_service import (
    viewer_cortex_mesh_pair,
)
from app.backend.services.viewer_service import viewer_metadata


COLORS = {
    "lh": (233, 184, 95, 230),  # Gold
    "rh": (82, 217, 225, 230),  # Cyan
}

AXES = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}


def _parse_original_pial_ply(
    payload: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse our verified position-only binary triangular PLY format."""

    marker = b"end_header\n"
    position = payload.find(marker)

    if position < 0 or position > 16384:
        raise ValueError("Invalid cortical PLY header.")

    header_end = position + len(marker)

    try:
        lines = payload[:header_end].decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("Invalid cortical PLY encoding.") from exc

    if (
        not lines
        or lines[0] != "ply"
        or "format binary_little_endian 1.0" not in lines
    ):
        raise ValueError("Unsupported cortical PLY format.")

    properties = [
        line for line in lines
        if line.startswith("property ")
    ]

    if properties != [
        "property float x",
        "property float y",
        "property float z",
        "property list uchar int vertex_indices",
    ]:
        raise ValueError("Unexpected cortical PLY vertex/face layout.")

    elements = [
        line for line in lines
        if line.startswith("element ")
    ]

    if len(elements) != 2:
        raise ValueError("Unexpected cortical PLY elements.")

    try:
        vertex_parts = elements[0].split()
        face_parts = elements[1].split()

        if (
            vertex_parts[:2] != ["element", "vertex"]
            or face_parts[:2] != ["element", "face"]
        ):
            raise ValueError("Unexpected PLY element ordering.")

        vertex_count = int(vertex_parts[2])
        face_count = int(face_parts[2])
    except (IndexError, ValueError) as exc:
        raise ValueError("Invalid cortical PLY counts.") from exc

    if not (
        0 < vertex_count <= 1_000_000
        and 0 < face_count <= 3_000_000
    ):
        raise ValueError("Cortical PLY counts outside expected limits.")

    face_dtype = np.dtype([
        ("count", "u1"),
        ("indices", "<i4", (3,)),
    ])

    vertices_size = vertex_count * 12
    faces_size = face_count * face_dtype.itemsize

    expected_size = header_end + vertices_size + faces_size

    if len(payload) != expected_size:
        raise ValueError("Invalid cortical PLY payload size.")

    vertices = np.frombuffer(
        payload,
        dtype="<f4",
        count=vertex_count * 3,
        offset=header_end,
    ).reshape(vertex_count, 3).astype(np.float64)

    records = np.frombuffer(
        payload,
        dtype=face_dtype,
        count=face_count,
        offset=header_end + vertices_size,
    )

    faces = records["indices"].astype(np.int64)

    if (
        not np.isfinite(vertices).all()
        or not np.all(records["count"] == 3)
        or np.any(faces < 0)
        or np.any(faces >= vertex_count)
    ):
        raise ValueError("Invalid cortical PLY geometry.")

    return vertices, faces


def viewer_cortex_contours_png(
    case_id: str,
    plane: str,
    index: int,
) -> bytes:
    """Return an RGBA MRI-aligned pial-contour PNG.

    Reuses the full registered-cortex identity, integrity and QC gate.
    Rendering geometry never authorizes a reconstruction by itself.
    """

    if plane not in AXES:
        raise ValueError("Unsupported anatomical plane.")

    if type(index) is not int:
        raise ValueError("Slice index must be an integer.")

    # Authorization occurs before reading or rendering any cortical mesh.
    meshes = viewer_cortex_mesh_pair(case_id)

    metadata = viewer_metadata(case_id)
    shape = tuple(int(v) for v in metadata["shape_ras_xyz"])

    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError("Invalid MRI geometry.")

    axis = AXES[plane]

    if index < 0 or index >= shape[axis]:
        raise ValueError("MRI slice index out of range.")

    affine = np.asarray(
        metadata["affine_ras"],
        dtype=np.float64,
    )

    width = shape[1] if plane == "sagittal" else shape[0]
    height = shape[1] if plane == "axial" else shape[2]

    # Supersampling makes narrow lines smoother when downsampled.
    scale = 2

    canvas = Image.new(
        "RGBA",
        (width * scale, height * scale),
        (0, 0, 0, 0),
    )

    painter = ImageDraw.Draw(canvas)

    for hemi in ("lh", "rh"):
        vertices_world, faces = _parse_original_pial_ply(
            meshes[hemi]
        )

        voxels = world_to_voxel(
            vertices_world,
            affine,
        )

        segments = intersect_slice(
            voxels,
            faces,
            axis=axis,
            index=index,
        )

        projected = project_to_image(
            segments,
            plane=plane,
            shape_ras_xyz=shape,
        )

        for segment in projected:
            points = [
                (
                    round(float(point[0]) * scale),
                    round(float(point[1]) * scale),
                )
                for point in segment
            ]

            painter.line(
                points,
                fill=COLORS[hemi],
                width=2,
            )

    canvas = canvas.resize(
        (width, height),
        resample=Image.Resampling.LANCZOS,
    )

    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()
