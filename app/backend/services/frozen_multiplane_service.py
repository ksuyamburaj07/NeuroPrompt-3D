"""Read-only coronal and sagittal previews of frozen MRI evidence.

Uses original MRI voxels and authenticated frozen predictions.
No inference, interpolation, metric calculation or source rewriting.
"""

from io import BytesIO

import nibabel as nib
import numpy as np
from PIL import Image

from app.backend.services.frozen_display_contrast import (
    display_grayscale,
)
from app.backend.services.frozen_plane_geometry import (
    PLANE_COUNTS,
    frozen_mask_plane,
    native_mri_plane,
)
from app.backend.services.frozen_test_service import (
    FrozenEvidenceError,
    load_verified_evaluation,
)
from app.backend.services.frozen_slice_service import (
    CANONICAL_AFFINE,
    MASK_KEYS,
    MRI_MANIFEST,
    MRI_MANIFEST_SHA256,
    MRI_ROOT,
    MODALITIES,
    OVERLAY_COLORS,
    FrozenSliceError,
    _load_mask_arrays,
    _validate_case_mri_headers,
    _verified_bytes,
    _frozen_mri_volume_contrast_window,
)


def render_frozen_nonaxial_png(
    case_index: int,
    plane: str,
    index: int,
    *,
    modality: str = "t2f",
    layer: str = "removed",
    contrast: str = "slice",
) -> bytes:
    """Render one authenticated native-LAS coronal or sagittal PNG."""

    if (
        type(case_index) is not int
        or not 1 <= case_index <= 129
        or plane not in ("coronal", "sagittal")
        or type(index) is not int
        or not 0 <= index < PLANE_COUNTS[plane]
        or modality not in MODALITIES
        or layer not in ("none", *MASK_KEYS)
        or contrast not in ("slice", "volume")
    ):
        raise ValueError("Invalid frozen anatomical slice request.")

    try:
        evaluation = load_verified_evaluation()
    except FrozenEvidenceError as exc:
        raise FrozenSliceError(
            "Verified evaluation unavailable."
        ) from exc

    # Verifies the original archive, per-case mask hash,
    # identity, geometry and removal-only invariants.
    case_id, masks = _load_mask_arrays(
        case_index, evaluation
    )

    # Enforce four-modality geometry agreement, including the
    # historically documented unknown-unit exception rule.
    _validate_case_mri_headers(case_id)

    manifest_raw = _verified_bytes(
        MRI_MANIFEST,
        MRI_MANIFEST_SHA256,
    )

    try:
        entries = {}

        for line in manifest_raw.decode("utf-8").splitlines():
            digest, relative = line.split(maxsplit=1)

            if relative in entries:
                raise FrozenSliceError(
                    "Duplicate frozen MRI manifest entry."
                )

            entries[relative] = digest

    except (UnicodeError, ValueError) as exc:
        raise FrozenSliceError(
            "Frozen MRI manifest is invalid."
        ) from exc

    relative = f"{case_id}/{case_id}-{modality}.nii.gz"

    if len(entries) != 516 or relative not in entries:
        raise FrozenSliceError(
            "Requested MRI is absent from the frozen manifest."
        )

    mri_path = MRI_ROOT / relative

    # Authenticate the original MRI bytes before decoding.
    _verified_bytes(mri_path, entries[relative])

    try:
        image = nib.load(str(mri_path))

        if (
            image.shape != (182, 218, 182)
            or not np.allclose(
                image.affine,
                CANONICAL_AFFINE,
                rtol=0,
                atol=1e-5,
            )
        ):
            raise FrozenSliceError(
                "Frozen MRI geometry mismatch."
            )

        # Native MRI XYZ -> display rows/columns.
        gray_plane = np.asarray(
            native_mri_plane(
                image.dataobj,
                plane,
                index,
            ),
            dtype=np.float32,
        )

    except FrozenSliceError:
        raise

    except Exception as exc:
        raise FrozenSliceError(
            "Original MRI plane could not be decoded."
        ) from exc

    if not np.isfinite(gray_plane).all():
        raise FrozenSliceError(
            "Original MRI plane contains invalid intensities."
        )

    if contrast == "volume":
        window = _frozen_mri_volume_contrast_window(
            case_id, modality
        )
        grayscale = display_grayscale(
            gray_plane, window
        )
    else:
        # Visualization-only contrast scaling.
        foreground = gray_plane[gray_plane != 0]

        if foreground.size:
            lo, hi = np.percentile(foreground, (1, 99))
        else:
            lo, hi = 0.0, 0.0

        if hi > lo:
            grayscale = (
                np.clip(
                    (gray_plane - lo) / (hi - lo),
                    0.0,
                    1.0,
                ) * 255.0
            ).astype(np.uint8)
        else:
            grayscale = np.zeros_like(
                gray_plane,
                dtype=np.uint8,
            )

    rendered = Image.fromarray(grayscale).convert("RGBA")

    if layer != "none":
        # Frozen ZYX mask -> exactly the same display grid.
        mask_plane = frozen_mask_plane(
            masks[layer],
            plane,
            index,
        )

        if mask_plane.shape != grayscale.shape:
            raise FrozenSliceError(
                "Frozen MRI and mask plane mismatch."
            )

        overlay = np.zeros(
            (*mask_plane.shape, 4),
            dtype=np.uint8,
        )

        overlay[mask_plane] = OVERLAY_COLORS[layer]

        rendered = Image.alpha_composite(
            rendered,
            Image.fromarray(overlay, mode="RGBA"),
        )

    output = BytesIO()

    rendered.convert("RGB").save(
        output,
        format="PNG",
    )

    return output.getvalue()
