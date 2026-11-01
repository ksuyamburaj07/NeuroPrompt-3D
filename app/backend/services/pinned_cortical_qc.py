"""Local-only genuine MRI + pial-contour research QC service.

PROVISIONAL anatomical QC. No production case access, approval,
inference, model mutation, or source-file modification.
"""

import hashlib
import os
import json
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from app.backend.services.cortical_slice_geometry import (
    intersect_slice,
    project_to_image,
    world_to_voxel,
)
from app.backend.services.viewer_cortex_contour_service import (
    _parse_original_pial_ply,
)
from app.backend.services.viewer_service import _oriented_slice


ROOT = Path(
    os.environ.get(
        "NP3D_QC_RECONSTRUCTION_ROOT",
        "/data/NeuroPrompt3D_Experiments/"
        "cortical_reconstruction/m11e6b5_trial01",
    )
)
MRI = Path(
    os.environ.get(
        "NP3D_QC_MRI_FILE",
        "/data/NeuroPrompt3D_Experiments/"
        "cortical_feasibility/synthseg_trial/"
        "input_t1n_original.nii.gz",
    )
)
ASSETS = ROOT / "qc_review/np3d-b7d-real-qc"
EXPORTS = ROOT / "viewer_exports"

EXPECTED_MRI_SHA256 = (
    "4f27af95641365549b4af215aa210bd30a86ff1a00aa7281010daacaa235a2dc"
)

PLANES = {"sagittal": 0, "coronal": 1, "axial": 2}
COLORS = {
    "lh": (233, 184, 95, 230),
    "rh": (82, 217, 225, 230),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize():
    source_hash = sha256(MRI)
    if source_hash != EXPECTED_MRI_SHA256:
        raise RuntimeError(
            "STOP: Original T1N fingerprint does not match "
            "the recorded reconstruction source."
        )

    source = nib.load(str(MRI))
    canonical = nib.as_closest_canonical(source)

    if nib.aff2axcodes(canonical.affine) != ("R", "A", "S"):
        raise RuntimeError("Unexpected canonical MRI orientation.")

    volume = canonical.get_fdata(dtype=np.float32)
    shape = tuple(map(int, volume.shape))

    if len(shape) != 3 or not np.isfinite(volume).all():
        raise RuntimeError("Invalid original MRI volume.")

    foreground = volume[volume > 0]
    if foreground.size == 0:
        raise RuntimeError("Original MRI has no positive foreground.")

    low, high = map(
        float, np.percentile(foreground, [1, 99.5])
    )
    if not high > low:
        raise RuntimeError("Invalid MRI intensity range.")

    surfaces = {}
    for hemi in ("lh", "rh"):
        filename = f"{hemi}.pial.ras.ply"
        preview_file = ASSETS / filename
        verified_export = EXPORTS / filename

        if sha256(preview_file) != sha256(verified_export):
            raise RuntimeError(
                f"STOP: {hemi} QC mesh differs from verified export."
            )

        vertices, faces = _parse_original_pial_ply(
            preview_file.read_bytes()
        )
        surfaces[hemi] = (
            world_to_voxel(vertices, canonical.affine),
            faces,
        )

    return volume, shape, low, high, surfaces, source_hash, canonical.affine.tolist()


(
    VOLUME,
    SHAPE,
    LOW,
    HIGH,
    SURFACES,
    SOURCE_HASH,
    AFFINE_RAS,
) = initialize()


def image_size(plane):
    x, y, z = SHAPE
    if plane == "axial":
        return x, y
    if plane == "coronal":
        return x, z
    return y, z



def exterior_zero_mask(image):
    """Mask exterior source zeros, preserving enclosed dark voxels."""
    zeros = image == 0
    height, width = zeros.shape
    outside = np.zeros_like(zeros, dtype=bool)

    stack = []
    for x in range(width):
        stack.extend(((0, x), (height - 1, x)))
    for y in range(height):
        stack.extend(((y, 0), (y, width - 1)))

    while stack:
        y, x = stack.pop()
        if (
            y < 0 or x < 0 or
            y >= height or x >= width or
            outside[y, x] or not zeros[y, x]
        ):
            continue

        outside[y, x] = True
        stack.extend((
            (y - 1, x), (y + 1, x),
            (y, x - 1), (y, x + 1)
        ))

    return outside


@lru_cache(maxsize=24)
def render_png(kind, plane, index):
    axis = PLANES[plane]
    width, height = image_size(plane)

    if kind in ("slices", "slices3d"):
        image = _oriented_slice(VOLUME, plane, index)
        pixels = np.uint8(
            np.rint(
                np.clip((image - LOW) / (HIGH - LOW), 0, 1)
                * 255
            )
        )
        output_image = Image.fromarray(pixels)

        if kind == "slices3d":
            outside = exterior_zero_mask(image)
            rgba = np.empty((*pixels.shape, 4), dtype=np.uint8)
            rgba[..., :3] = pixels[..., None]
            rgba[..., 3] = np.where(outside, 0, 255)
            output_image = Image.fromarray(rgba)

    else:
        scale = 2
        canvas = Image.new(
            "RGBA",
            (width * scale, height * scale),
            (0, 0, 0, 0),
        )
        painter = ImageDraw.Draw(canvas)

        for hemi in ("lh", "rh"):
            vertices, faces = SURFACES[hemi]
            segments = intersect_slice(
                vertices, faces, axis, index
            )
            projected = project_to_image(
                segments, plane, SHAPE
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

        output_image = canvas.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

    buffer = BytesIO()
    output_image.save(buffer, format="PNG")
    return buffer.getvalue()


def metadata():
    x, y, z = SHAPE
    return {
        "orientation_convention": "RAS+",
        "shape_ras_xyz": [x, y, z],
        "affine_ras": AFFINE_RAS,
        "source_sha256": SOURCE_HASH,
        "reconstruction_id": "m11e6b5_trial01",
        "qc_status": "provisional",
        "clinically_validated": False,
        "planes": {
            "axial": {
                "axis": 2, "count": z,
                "center_index": z // 2,
                "width": x, "height": y,
            },
            "coronal": {
                "axis": 1, "count": y,
                "center_index": y // 2,
                "width": x, "height": z,
            },
            "sagittal": {
                "axis": 0, "count": x,
                "center_index": x // 2,
                "width": y, "height": z,
            },
        },
    }
