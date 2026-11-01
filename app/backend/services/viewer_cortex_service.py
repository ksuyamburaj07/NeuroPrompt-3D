"""Read-only cortex access bound to exact validated staged T1N bytes.

No filename, geometry, application UUID or inference output identifies a
reconstruction. Every request rechecks the source and both registered meshes.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.backend.core import paths
from app.backend.schemas.cortex import CortexRegistry
from app.backend.services.case_service import load_live_case


class CortexUnavailable(FileNotFoundError):
    """The optional reconstruction is not eligible for display."""


@dataclass
class _Evaluation:
    status: dict
    mesh: bytes | None = None
    meshes: dict[str, bytes] | None = None


def _confined_file(root: Path, relative: Path) -> Path:
    """Accept regular files under one storage root, without symbolic links."""
    if (
        any(parent.is_symlink() for parent in (root, *root.parents))
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError("Unsafe cortical asset path.")

    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Symbolic links are not permitted.")

    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("File escapes its storage root.")
    if not resolved.is_file():
        raise FileNotFoundError("Required file is unavailable.")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate(case_id: str, hemisphere: str | None = None) -> _Evaluation:
    # Retain the M11 invalid-ID / missing-case contract even when disabled.
    case = load_live_case(case_id)
    status = {
        "case_id": case_id,
        "available": False,
        "reason": "case_not_ready",
        "reconstruction_id": None,
        "coordinate_convention": "physical RAS+",
        "units": "mm",
        "qc": None,
    }
    result = _Evaluation(status)
    if (
        case.case_id != case_id
        or not case.valid
        or case.status != "ready"
        or not case.ready_for_inference
        or "t1n" not in case.modalities
        or case.modalities["t1n"].valid is not True
    ):
        return result

    registry_path = paths.CORTEX_REGISTRY_PATH
    if registry_path is None:
        status["reason"] = "registry_disabled"
        return result

    try:
        registry = CortexRegistry.model_validate_json(
            registry_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        status["reason"] = "registry_missing"
        return result
    except (OSError, ValueError, RuntimeError):
        status["reason"] = "registry_invalid"
        return result

    if not registry.enabled:
        status["reason"] = "registry_disabled"
        return result

    try:
        _confined_file(paths.LIVE_CASES_ROOT, Path(case_id) / "case.json")
        root = paths.LIVE_CASES_ROOT / case_id / "modalities"
        # Ambiguous staged sources fail closed instead of choosing a suffix.
        candidates = [
            root / f"t1n{suffix}" for suffix in (".nii.gz", ".nii")
            if (root / f"t1n{suffix}").exists()
            or (root / f"t1n{suffix}").is_symlink()
        ]
        if len(candidates) != 1:
            raise ValueError("Exactly one validated source is required.")
        source = _confined_file(
            paths.LIVE_CASES_ROOT,
            candidates[0].relative_to(paths.LIVE_CASES_ROOT),
        )
        source_sha256 = _sha256(source)
    except (OSError, ValueError, RuntimeError):
        status["reason"] = "source_unavailable"
        return result

    # Legacy cases have no validation-time source fingerprint.
    # They remain usable in the standard MRI viewer, but
    # cannot access case-specific cortical reconstructions.
    if case.source_t1n_sha256 is None:
        status["reason"] = "source_unavailable"
        return result

    # Never select a new reconstruction if the staged MRI
    # has changed since successful case validation.
    if source_sha256 != case.source_t1n_sha256:
        status["reason"] = "source_identity_mismatch"
        return result

    entry = next(
        (item for item in registry.entries
         if item.source_t1n_sha256 == source_sha256),
        None,
    )
    if entry is None:
        status["reason"] = "source_unregistered"
        return result

    status["reconstruction_id"] = entry.reconstruction_id
    status["qc"] = entry.qc.model_dump()

    try:
        asset_root = Path(entry.asset_root)
        # Medical assets must stay outside this Git checkout.
        if asset_root.resolve().is_relative_to(paths.PROJECT_ROOT.resolve()):
            raise ValueError("Cortical assets must use external storage.")
        selected_mesh = None
        verified_meshes = {}
        for side in ("lh", "rh"):
            asset = entry.meshes[side]
            path = _confined_file(asset_root, Path(asset.path))
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != asset.sha256:
                status["reason"] = "asset_integrity_failed"
                return result
            verified_meshes[side] = payload
            if side == hemisphere:
                selected_mesh = payload
    except (OSError, ValueError, RuntimeError):
        status["reason"] = "assets_unavailable"
        return result

    if entry.qc.status != "approved":
        status["reason"] = "qc_not_approved"
        return result

    status["available"] = True
    status["reason"] = "available"
    # Return the very bytes hashed above, never reopen a checked mesh path.
    result.mesh = selected_mesh
    result.meshes = verified_meshes
    return result


def viewer_cortex_status(case_id: str) -> dict:
    return _evaluate(case_id).status


def viewer_cortex_ply(case_id: str, hemisphere: str) -> bytes:
    if hemisphere not in {"lh", "rh"}:
        raise ValueError("Unsupported cortical hemisphere; use lh or rh.")

    result = _evaluate(case_id, hemisphere)
    if not result.status["available"] or result.mesh is None:
        raise CortexUnavailable("Cortical reconstruction is unavailable.")
    return result.mesh



def viewer_cortex_mesh_pair(case_id: str) -> dict[str, bytes]:
    """Return both verified hemispheres from one eligibility evaluation.

    Reuses the full M11 cortical identity/integrity/QC gate.
    No path or registered asset is exposed when unavailable.
    """
    result = _evaluate(case_id)

    if (
        not result.status["available"]
        or result.meshes is None
        or set(result.meshes) != {"lh", "rh"}
    ):
        raise CortexUnavailable(
            "Cortical reconstruction is unavailable."
        )

    return dict(result.meshes)
