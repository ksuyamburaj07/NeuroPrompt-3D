"""Read-only spatial marker metadata for completed live MRI runs."""

import nibabel as nib
import numpy as np

from app.backend.services.artifact_service import ArtifactRunNotComplete
from app.backend.services.run_service import load_live_run
from app.backend.services.viewer_service import _modality_path


class ViewerRunCaseMismatch(Exception):
    """An inference run does not belong to the requested case."""


class ViewerCoordinateError(Exception):
    """A recorded coordinate cannot be represented safely in the MRI."""


def marker_to_ras(
    source_zyx: list[int] | tuple[int, int, int],
    image: nib.spatialimages.SpatialImage,
) -> dict:
    """Convert native tensor ZYX to canonical RAS XYZ using affines."""

    if len(source_zyx) != 3:
        raise ViewerCoordinateError("Marker must contain three coordinates.")

    if len(image.shape) != 3:
        raise ViewerCoordinateError("Source MRI must be three-dimensional.")

    native_xyz = np.asarray(
        [
            source_zyx[2],
            source_zyx[1],
            source_zyx[0],
        ],
        dtype=np.float64,
    )

    if (
        not np.isfinite(native_xyz).all()
        or not np.equal(native_xyz, np.rint(native_xyz)).all()
        or np.any(native_xyz < 0)
        or np.any(native_xyz >= np.asarray(image.shape))
    ):
        raise ViewerCoordinateError(
            "Recorded marker lies outside native MRI voxel geometry."
        )

    canonical = nib.as_closest_canonical(image)

    native_homogeneous = np.append(native_xyz, 1.0)

    world = np.asarray(
        image.affine, dtype=np.float64
    ) @ native_homogeneous

    try:
        canonical_voxel = np.linalg.solve(
            np.asarray(canonical.affine, dtype=np.float64),
            world,
        )[:3]
    except np.linalg.LinAlgError as exc:
        raise ViewerCoordinateError(
            "Source MRI affine is not invertible."
        ) from exc

    rounded = np.rint(canonical_voxel)

    if (
        not np.isfinite(canonical_voxel).all()
        or not np.allclose(
            canonical_voxel,
            rounded,
            rtol=0,
            atol=1e-4,
        )
        or np.any(rounded < 0)
        or np.any(rounded >= np.asarray(canonical.shape))
    ):
        raise ViewerCoordinateError(
            "Marker cannot be mapped to a valid RAS voxel."
        )

    return {
        "source_zyx": [int(value) for value in source_zyx],
        "ras_xyz": [int(value) for value in rounded],
        "world_ras_mm": [float(value) for value in world[:3]],
    }


def viewer_run_markers(case_id: str, run_id: str) -> dict:
    run = load_live_run(run_id)

    if run.case_id != case_id:
        raise ViewerRunCaseMismatch(
            "Inference run belongs to a different MRI case."
        )

    if run.status != "complete":
        raise ArtifactRunNotComplete(
            "Scientific markers are available after run completion."
        )

    source_path = _modality_path(case_id, "t1n")
    image = nib.load(str(source_path))

    hotspot = (
        marker_to_ras(run.hotspot_zyx, image)
        if run.hotspot_zyx is not None
        else None
    )

    negative_prompt = (
        marker_to_ras(run.fp_prompt_zyx, image)
        if run.fp_prompt_zyx is not None
        else None
    )

    return {
        "case_id": case_id,
        "run_id": run_id,
        "orientation_convention": "RAS+",
        "action": run.action,
        "gate_state": run.gate_state,
        "hotspot_variance": run.hotspot_variance,
        "hotspot": hotspot,
        "negative_prompt": negative_prompt,
        "sam_used": run.sam_used,
    }
