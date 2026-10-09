"""Read-only rendering of verified live-run segmentation masks."""

from functools import lru_cache
from io import BytesIO

import nibabel as nib
import numpy as np
from PIL import Image

from app.backend.services.artifact_service import (
    resolve_live_artifact,
)
from app.backend.services.run_service import load_live_run
from app.backend.services.viewer_service import (
    MAX_VOXELS,
    PLANE_AXES,
    _modality_path,
)


LAYERS = {
    "baseline": ("baseline_mask_nifti", (255, 95, 108)),
    "final": ("final_mask_nifti", (68, 224, 157)),
    "removed": ("removal_mask_nifti", (255, 182, 69)),
}


class OverlayGeometryMismatch(Exception):
    """The saved mask cannot be safely aligned with the MRI."""


class OverlayCaseMismatch(Exception):
    """The run belongs to a different staged case."""


@lru_cache(maxsize=2)
def _aligned_mask(
    mask_filename: str,
    mask_mtime_ns: int,
    reference_filename: str,
    reference_mtime_ns: int,
) -> np.ndarray:
    """Load a binary display mask after verifying world-space geometry."""

    # Modification timestamps are part of the cache identity.
    del mask_mtime_ns, reference_mtime_ns

    reference = nib.load(reference_filename)
    mask = nib.load(mask_filename)

    if len(reference.shape) != 3 or len(mask.shape) != 3:
        raise OverlayGeometryMismatch(
            "MRI and segmentation must both be 3D volumes."
        )

    if (
        int(np.prod(reference.shape, dtype=np.int64)) > MAX_VOXELS
        or int(np.prod(mask.shape, dtype=np.int64)) > MAX_VOXELS
    ):
        raise OverlayGeometryMismatch(
            "Segmentation exceeds the viewer's volume limit."
        )

    reference_ras = nib.as_closest_canonical(reference)
    mask_ras = nib.as_closest_canonical(mask)

    if (
        mask_ras.shape != reference_ras.shape
        or not np.allclose(
            mask_ras.affine,
            reference_ras.affine,
            rtol=0,
            atol=1e-5,
        )
    ):
        raise OverlayGeometryMismatch(
            "Segmentation geometry does not match the source MRI."
        )

    values = np.asanyarray(mask_ras.dataobj)

    if not np.isfinite(values).all():
        raise OverlayGeometryMismatch(
            "Segmentation contains non-finite voxel values."
        )

    # This is visualization only. The source mask is unchanged.
    return np.asarray(values != 0, dtype=np.uint8)


def overlay_slice_png(
    case_id: str,
    run_id: str,
    layer: str,
    plane: str,
    index: int,
) -> bytes:
    """Produce an anatomically aligned transparent RGBA PNG."""

    if layer not in LAYERS:
        raise ValueError("Unsupported segmentation layer.")

    if plane not in PLANE_AXES:
        raise ValueError("Unsupported anatomical plane.")

    run = load_live_run(run_id)

    if run.case_id != case_id:
        raise OverlayCaseMismatch(
            "This inference run belongs to another case."
        )

    artifact_name, color = LAYERS[layer]

    # Reuse the approved live-artifact path and completion checks.
    artifact = resolve_live_artifact(
        run_id, artifact_name
    )

    reference_path = _modality_path(case_id, "t1n")
    mask_path = artifact.path

    mask = _aligned_mask(
        str(mask_path),
        mask_path.stat().st_mtime_ns,
        str(reference_path),
        reference_path.stat().st_mtime_ns,
    )

    axis = PLANE_AXES[plane]

    if index < 0 or index >= mask.shape[axis]:
        raise ValueError("Segmentation slice index is out of range.")

    # Exactly the same orientation transform as viewer_slice_png.
    pixels = np.ascontiguousarray(
        np.take(mask, index, axis=axis).T[::-1, :]
    )

    rgba = np.zeros(
        (*pixels.shape, 4),
        dtype=np.uint8,
    )

    rgba[pixels != 0] = (*color, 255)

    output = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(
        output, format="PNG"
    )
    return output.getvalue()
