"""Authenticated, read-only axial preview of frozen test predictions.

The original MRI, prediction archive and evaluation are never changed.
Output is a visual PNG, not a new scientific prediction or metric.
"""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import re
import tarfile

import nibabel as nib
import numpy as np

from app.backend.services.frozen_display_contrast import (
    display_grayscale,
    verified_volume_contrast_window,
)
from PIL import Image

from app.backend.services.frozen_test_service import (
    load_verified_evaluation,
    FrozenEvidenceError,
)
from app.backend.core.paths import FROZEN_M9E_ROOT


class FrozenSliceError(RuntimeError):
    """Required frozen image evidence is unavailable or inconsistent."""


PREDICTION_ROOT = (
    FROZEN_M9E_ROOT / "2026-09-29"
    / "M9E3_PRE_GT_PREDICTION_FREEZE"
)

PREDICTION_MANIFEST = (
    PREDICTION_ROOT
    / "M9E3_PREDICTION_MANIFEST_2026_09_29.json"
)

PREDICTION_ARCHIVE = (
    PREDICTION_ROOT
    / "M9E3_PRE_GT_PREDICTIONS_2026_09_29.tar.gz"
)

MRI_MANIFEST = (
    FROZEN_M9E_ROOT / "2026-09-27"
    / "M9E2_PRE_GT"
    / "M9E2_TEST_MRI_SHA256_2026_09_27.txt"
)

MRI_MANIFEST_SHA256 = (
    "13cb4ec43eec2c90208bddd09d139c655ec8b438a"
    "21af29b04a42968be11dbdf"
)

MRI_ROOT = Path(
    "/data/Datasets/Raw/BraTS2024/"
    "AdultGliomaPostTreatment/"
    "M9E_TEST_MRI_ONLY/training_data1_v2"
)

CANONICAL_AFFINE = np.array([
    [-1, 0, 0, 90],
    [0, 1, 0, -126],
    [0, 0, 1, -72],
    [0, 0, 0, 1],
], dtype=np.float64)

MASK_KEYS = {
    "baseline": "baseline_mask_zyx",
    "final": "final_mask_zyx",
    "removed": "removal_mask_zyx",
    "support": "support_mask_zyx",
}

OVERLAY_COLORS = {
    "baseline": (30, 200, 190, 130),
    "final": (176, 125, 250, 130),
    "removed": (255, 155, 45, 175),
    "support": (70, 145, 255, 90),
}

MODALITIES = {"t1n", "t1c", "t2w", "t2f"}


def _verified_bytes(path: Path, expected_sha: str) -> bytes:
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise FrozenSliceError("Frozen source is missing.") from exc

    if sha256(contents).hexdigest() != expected_sha:
        raise FrozenSliceError("Frozen source checksum mismatch.")

    return contents


def _case_record(document: dict, index: int) -> dict:
    records = [
        item for item in document["case_results"]
        if item["case_index"] == index
    ]

    if len(records) != 1:
        raise FrozenSliceError("Frozen evaluation case mismatch.")

    return records[0]


def _load_mask_arrays(
    index: int,
    evaluation: dict,
) -> tuple[str, dict[str, np.ndarray]]:
    source = evaluation["prediction_source"]

    manifest_raw = _verified_bytes(
        PREDICTION_MANIFEST,
        source["m9e3_prediction_manifest_sha256"],
    )

    archive_raw = _verified_bytes(
        PREDICTION_ARCHIVE,
        source["m9e3_archive_sha256"],
    )

    try:
        manifest = json.loads(manifest_raw)
        cases = manifest["cases"]

        if (
            len(cases) != 129
            or {item["case_index"] for item in cases}
            != set(range(1, 130))
        ):
            raise FrozenSliceError(
                "Prediction case inventory is inconsistent."
            )

        entries = [
            item for item in cases
            if item["case_index"] == index
        ]
        if len(entries) != 1:
            raise FrozenSliceError("Prediction case is missing.")

        record = entries[0]
        case_id = record["case_id"]
        evaluated = _case_record(evaluation, index)

        if (
            not isinstance(case_id, str)
            or not re.fullmatch(
                r"BraTS-GLI-\d{5}-\d{3}", case_id
            )
            or case_id != evaluated["case_id"]
            or record["policy"]["action"] != evaluated["action"]
            or tuple(record["geometry"]["shape_xyz"])
            != (182, 218, 182)
        ):
            raise FrozenSliceError(
                "Frozen case identity or geometry mismatch."
            )

        with tarfile.open(
            fileobj=BytesIO(archive_raw), mode="r:gz"
        ) as tar:
            members = [
                member for member in tar
                if member.isfile()
                and Path(member.name).parts[-2:]
                == (case_id, "PREDICTION_MASKS.npz")
            ]

            if len(members) != 1:
                raise FrozenSliceError(
                    "Frozen mask member is missing or duplicated."
                )

            member = members[0]
            if member.size > 16_000_000:
                raise FrozenSliceError(
                    "Frozen mask payload exceeds the safety limit."
                )

            stream = tar.extractfile(member)
            if stream is None:
                raise FrozenSliceError("Mask member is unreadable.")

            raw = stream.read()

        expected = record["file_sha256"]["PREDICTION_MASKS.npz"]

        if sha256(raw).hexdigest() != expected:
            raise FrozenSliceError(
                "Individual prediction checksum mismatch."
            )

        with np.load(
            BytesIO(raw), allow_pickle=False
        ) as saved:
            if set(saved.files) != set(MASK_KEYS.values()):
                raise FrozenSliceError(
                    "Unexpected frozen prediction arrays."
                )

            masks = {}
            for label, key in MASK_KEYS.items():
                arr = saved[key]
                if (
                    arr.shape != (182, 218, 182)
                    or arr.dtype != np.uint8
                    or np.any(arr > 1)
                ):
                    raise FrozenSliceError(
                        "Invalid frozen mask array."
                    )
                masks[label] = arr.astype(bool)

        baseline = masks["baseline"]
        final = masks["final"]
        removed = masks["removed"]
        support = masks["support"]

        if (
            np.any(final & ~baseline)
            or not np.array_equal(
                removed, baseline & ~final
            )
            or np.any(removed & ~support)
        ):
            raise FrozenSliceError(
                "Saved prediction policy invariants failed."
            )

        return case_id, masks

    except (
        KeyError, TypeError, ValueError, IndexError,
        tarfile.TarError, OSError,
    ) as exc:
        raise FrozenSliceError(
            "Frozen prediction evidence cannot be decoded."
        ) from exc



def _validate_case_mri_headers(case_id: str) -> None:
    """Enforce the frozen four-modality MRI geometry contract.

    Accept unknown spatial units only when all four modalities
    agree and each matches the canonical numerical geometry.
    No voxel arrays are loaded here.
    """
    if not re.fullmatch(
        r"BraTS-GLI-\d{5}-\d{3}", case_id
    ):
        raise FrozenSliceError("Invalid frozen MRI case identity.")

    reference_affine = None
    observed_units = set()

    for modality in ("t1n", "t1c", "t2w", "t2f"):
        path = (
            MRI_ROOT / case_id
            / f"{case_id}-{modality}.nii.gz"
        )

        if not path.is_file():
            raise FrozenSliceError(
                "Required MRI modality is unavailable."
            )

        try:
            image = nib.load(str(path))
            shape = tuple(image.shape)
            affine = np.asarray(
                image.affine, dtype=np.float64
            )
            zooms = np.asarray(
                image.header.get_zooms()[:3],
                dtype=np.float64,
            )
            units = image.header.get_xyzt_units()[0]
        except Exception as exc:
            raise FrozenSliceError(
                "Frozen MRI header cannot be decoded."
            ) from exc

        valid = (
            shape == (182, 218, 182)
            and affine.shape == (4, 4)
            and zooms.shape == (3,)
            and np.isfinite(affine).all()
            and np.isfinite(zooms).all()
            and np.array_equal(
                zooms, np.array([1.0, 1.0, 1.0])
            )
            and np.allclose(
                affine,
                CANONICAL_AFFINE,
                rtol=0,
                atol=1e-5,
            )
            and units in ("mm", "unknown")
        )

        if not valid:
            raise FrozenSliceError(
                "Frozen MRI geometry validation failed."
            )

        if reference_affine is not None:
            if not np.allclose(
                affine,
                reference_affine,
                rtol=0,
                atol=1e-5,
            ):
                raise FrozenSliceError(
                    "MRI modalities have inconsistent affines."
                )
        else:
            reference_affine = affine.copy()

        observed_units.add(units)

    if len(observed_units) != 1:
        raise FrozenSliceError(
            "MRI modalities have inconsistent spatial units."
        )


def _load_verified_mri_slice(
    case_id: str,
    modality: str,
    z: int,
) -> np.ndarray:
    manifest_raw = _verified_bytes(
        MRI_MANIFEST, MRI_MANIFEST_SHA256
    )

    relative_name = (
        f"{case_id}/{case_id}-{modality}.nii.gz"
    )

    entries = {}
    for line in manifest_raw.decode("utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        if filename in entries:
            raise FrozenSliceError("Duplicate MRI manifest entry.")
        entries[filename] = digest

    if len(entries) != 516 or relative_name not in entries:
        raise FrozenSliceError(
            "MRI input is absent from the frozen manifest."
        )

    file_path = MRI_ROOT / relative_name

    # Authenticate the original file before decoding any voxels.
    _verified_bytes(file_path, entries[relative_name])

    # Enforce the frozen four-modality input-validation contract.
    _validate_case_mri_headers(case_id)

    try:
        image = nib.load(str(file_path))

        if (
            image.shape != (182, 218, 182)
            or not np.allclose(
                image.affine,
                CANONICAL_AFFINE,
                rtol=0,
                atol=1e-5,
            )
            or not np.allclose(
                image.header.get_zooms()[:3],
                (1.0, 1.0, 1.0),
                rtol=0,
                atol=1e-5,
            )
            or image.header.get_xyzt_units()[0]
            not in ("mm", "unknown")
        ):
            raise FrozenSliceError(
                "MRI geometry does not match frozen input geometry."
            )

        native_xy = np.asarray(
            image.dataobj[:, :, z],
            dtype=np.float32,
        )

        if not np.isfinite(native_xy).all():
            raise FrozenSliceError(
                "MRI slice contains non-finite values."
            )

        return native_xy

    except (OSError, ValueError, IndexError) as exc:
        raise FrozenSliceError(
            "Original MRI slice cannot be decoded."
        ) from exc



def _frozen_mri_volume_contrast_window(
    case_id: str,
    modality: str,
) -> tuple[float, float]:
    """Return a source-authenticated whole-volume display window."""

    if (
        not re.fullmatch(r"BraTS-GLI-\d{5}-\d{3}", case_id)
        or modality not in MODALITIES
    ):
        raise FrozenSliceError("Invalid MRI source identity.")

    raw = _verified_bytes(
        MRI_MANIFEST, MRI_MANIFEST_SHA256
    )

    try:
        entries = {}
        for line in raw.decode("utf-8").splitlines():
            digest, relative = line.split(maxsplit=1)
            if relative in entries:
                raise FrozenSliceError(
                    "Duplicate MRI manifest entry."
                )
            entries[relative] = digest
    except (UnicodeError, ValueError) as exc:
        raise FrozenSliceError(
            "Frozen MRI manifest cannot be decoded."
        ) from exc

    relative = f"{case_id}/{case_id}-{modality}.nii.gz"

    if len(entries) != 516 or relative not in entries:
        raise FrozenSliceError(
            "MRI absent from authenticated manifest."
        )

    source_path = MRI_ROOT / relative

    try:
        return verified_volume_contrast_window(
            source_path,
            entries[relative],
            verify_file=_verified_bytes,
        )
    except Exception as exc:
        raise FrozenSliceError(
            "Verified MRI contrast unavailable."
        ) from exc


def render_frozen_axial_png(
    case_index: int,
    z: int,
    *,
    modality: str = "t2f",
    layer: str = "removed",
    contrast: str = "slice",
) -> bytes:
    """Return one authenticated axial preview PNG.

    Image columns: native X increasing (patient R -> L).
    Image rows: native Y decreasing (anterior -> posterior).
    The masks are restored from ZYX directly to this native grid.
    """
    if (
        type(case_index) is not int
        or not 1 <= case_index <= 129
        or type(z) is not int
        or not 0 <= z < 182
        or modality not in MODALITIES
        or layer not in ("none", *MASK_KEYS)
        or contrast not in ("slice", "volume")
    ):
        raise ValueError("Invalid frozen slice request.")

    try:
        evaluation = load_verified_evaluation()
    except FrozenEvidenceError as exc:
        raise FrozenSliceError(
            "Frozen evaluation is unavailable."
        ) from exc

    case_id, masks = _load_mask_arrays(
        case_index, evaluation
    )

    native_xy = _load_verified_mri_slice(
        case_id, modality, z
    )

    if contrast == "volume":
        window = _frozen_mri_volume_contrast_window(
            case_id, modality
        )
        gray_xy = display_grayscale(native_xy, window)
    else:
        # View-only intensity scaling, never scientific preprocessing.
        foreground = native_xy[native_xy != 0]

        if foreground.size:
            lo, hi = np.percentile(foreground, [1, 99])
        else:
            lo, hi = 0.0, 0.0

        if hi > lo:
            gray_xy = (
                np.clip(
                    (native_xy - lo) / (hi - lo),
                    0.0, 1.0,
                ) * 255.0
            ).astype(np.uint8)
        else:
            gray_xy = np.zeros_like(
                native_xy, dtype=np.uint8
            )

    # Native XYZ MRI -> display (Y reversed, X increasing).
    image_yx = np.flipud(gray_xy.T)

    rendered = Image.fromarray(image_yx).convert("RGBA")

    if layer != "none":
        # Frozen ZYX mask slice is already [Y, X].
        overlay_yx = np.flipud(masks[layer][z, :, :])

        if overlay_yx.shape != image_yx.shape:
            raise FrozenSliceError(
                "MRI and mask display geometry mismatch."
            )

        pixels = np.zeros(
            (*overlay_yx.shape, 4), dtype=np.uint8
        )
        pixels[overlay_yx] = OVERLAY_COLORS[layer]

        rendered = Image.alpha_composite(
            rendered,
            Image.fromarray(pixels, mode="RGBA"),
        )

    buffer = BytesIO()
    rendered.convert("RGB").save(
        buffer, format="PNG"
    )

    return buffer.getvalue()
