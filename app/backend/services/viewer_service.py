"""Read-only, RAS-oriented MRI visualization for staged live cases."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from app.backend.core import paths
from app.backend.services.case_service import load_live_case


MODALITIES = frozenset({"t1n", "t1c", "t2w", "t2f"})
PLANE_AXES = {"axial": 2, "coronal": 1, "sagittal": 0}

# Limits visualization memory use on the local development laptop.
MAX_VOXELS = 20_000_000


def _modality_path(case_id: str, modality: str) -> Path:
    if modality not in MODALITIES:
        raise ValueError("Unsupported MRI modality.")

    case = load_live_case(case_id)

    if not case.valid or not case.ready_for_inference:
        raise ValueError("Case is not ready for MRI visualization.")

    root = paths.LIVE_CASES_ROOT / case_id / "modalities"

    for suffix in (".nii.gz", ".nii"):
        candidate = root / f"{modality}{suffix}"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError("Validated MRI modality is unavailable.")


@lru_cache(maxsize=2)
def _load_ras(path_string: str, file_mtime_ns: int):
    """Cache at most two 8-bit visualization volumes.

    The timestamp participates in cache identity. Original NIfTI
    files and their scientific voxel values remain unchanged.
    """
    del file_mtime_ns

    image = nib.load(path_string)

    if len(image.shape) != 3:
        raise ValueError("Viewer requires a three-dimensional MRI.")

    if int(np.prod(image.shape, dtype=np.int64)) > MAX_VOXELS:
        raise ValueError("MRI exceeds the visualization voxel limit.")

    original_orientation = list(
        nib.aff2axcodes(image.affine)
    )

    canonical = nib.as_closest_canonical(image)

    voxels = np.array(
        canonical.get_fdata(dtype=np.float32),
        copy=True,
    )

    finite = np.isfinite(voxels)
    foreground = finite & (voxels != 0)

    grayscale = np.zeros(voxels.shape, dtype=np.uint8)

    if np.any(foreground):
        low, high = np.percentile(
            voxels[foreground],
            [1, 99],
        )

        if high > low:
            voxels -= np.float32(low)
            voxels /= np.float32(high - low)
            np.clip(voxels, 0, 1, out=voxels)
            voxels *= np.float32(255)
            np.nan_to_num(
                voxels,
                copy=False,
                nan=0,
                posinf=255,
                neginf=0,
            )
            grayscale = voxels.astype(np.uint8)
            grayscale[~foreground] = 0
        else:
            grayscale[foreground] = 255

    return (
        grayscale,
        canonical.affine.tolist(),
        original_orientation,
    )


def _case_volume(case_id: str, modality: str):
    path = _modality_path(case_id, modality)
    return _load_ras(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=2)
def _load_source_foreground(
    path_string: str,
    file_mtime_ns: int,
) -> np.ndarray:
    """Visualization-only finite, nonzero MRI foreground."""
    del file_mtime_ns

    image = nib.load(path_string)

    if len(image.shape) != 3:
        raise ValueError("Viewer requires a three-dimensional MRI.")

    if int(np.prod(image.shape, dtype=np.int64)) > MAX_VOXELS:
        raise ValueError("MRI exceeds the visualization voxel limit.")

    canonical = nib.as_closest_canonical(image)
    voxels = canonical.get_fdata(dtype=np.float32)

    return np.isfinite(voxels) & (voxels != 0)


def _case_source_foreground(
    case_id: str,
    modality: str,
) -> np.ndarray:
    path = _modality_path(case_id, modality)
    return _load_source_foreground(
        str(path),
        path.stat().st_mtime_ns,
    )


def viewer_metadata(case_id: str) -> dict:
    volume, affine, original_orientation = _case_volume(
        case_id, "t1n"
    )

    x, y, z = map(int, volume.shape)

    return {
        "case_id": case_id,
        "orientation_convention": "RAS+",
        "original_axis_codes": original_orientation,
        "shape_ras_xyz": [x, y, z],
        "affine_ras": affine,
        "voxel_spacing_ras_mm": [
            float(value)
            for value in nib.affines.voxel_sizes(
                np.asarray(affine)
            )
        ],
        "display_intensity": "nonzero_volume_percentile_1_99",
        "planes": {
            "axial": {
                "axis": 2,
                "count": z,
                "center_index": z // 2,
                "width": x,
                "height": y,
            },
            "coronal": {
                "axis": 1,
                "count": y,
                "center_index": y // 2,
                "width": x,
                "height": z,
            },
            "sagittal": {
                "axis": 0,
                "count": x,
                "center_index": x // 2,
                "width": y,
                "height": z,
            },
        },
    }


def _oriented_slice(
    volume: np.ndarray,
    plane: str,
    index: int,
) -> np.ndarray:
    """Use the same RAS display orientation for intensity and alpha."""
    if plane == "axial":
        result = volume[:, :, index].T[::-1, :]
    elif plane == "coronal":
        result = volume[:, index, :].T[::-1, :]
    else:
        result = volume[index, :, :].T[::-1, :]

    return np.ascontiguousarray(result)


def viewer_slice_png(
    case_id: str,
    modality: str,
    plane: str,
    index: int,
    transparent_background: bool = False,
) -> bytes:
    if plane not in PLANE_AXES:
        raise ValueError("Unsupported anatomical plane.")

    volume, _, _ = _case_volume(case_id, modality)

    axis = PLANE_AXES[plane]

    if index < 0 or index >= volume.shape[axis]:
        raise ValueError("MRI slice index is out of range.")

    image_array = _oriented_slice(volume, plane, index)

    output = BytesIO()

    if transparent_background:
        foreground = _case_source_foreground(
            case_id, modality
        )

        if foreground.shape != volume.shape:
            raise ValueError(
                "MRI foreground geometry does not match display volume."
            )

        alpha = (
            _oriented_slice(foreground, plane, index)
            .astype(np.uint8) * 255
        )

        # Preserve every display grayscale value, including dark
        # foreground pixels. Only the alpha channel changes.
        image = Image.fromarray(image_array).convert("RGBA")
        image.putalpha(Image.fromarray(alpha))
    else:
        image = Image.fromarray(image_array)

    image.save(output, format="PNG")
    return output.getvalue()
