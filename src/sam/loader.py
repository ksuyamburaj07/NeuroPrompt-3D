"""Provenance-gated, CPU-only SAM-Med3D loading; no data or inference.

Frozen identifiers come from M9A1_Source_and_Checkpoint_Provenance.
Only the explicitly supplied source repository and checkpoint are accessed.
Call in a fresh process if another segment_anything or utils is imported.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

import torch

FROZEN_SAM_COMMIT = "f3de1fa10da98e46f49f176773d2b1e306ba131f"
FROZEN_TURBO_SHA256 = "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c"
SAM_REGISTRY_KEY = "vit_b_ori"


class SAMLoaderIntegrityError(ValueError):
    """Source, checkpoint, or model does not satisfy the frozen contract."""


@dataclass(frozen=True)
class SAMCheckpointBundle:
    model: torch.nn.Module
    source_root: Path
    source_commit: str
    checkpoint_path: Path
    checkpoint_sha256: str
    registry_key: str = SAM_REGISTRY_KEY


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(root), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise SAMLoaderIntegrityError("Cannot verify SAM Git repository") from exc


def _verify_source(root: Path) -> str:
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise SAMLoaderIntegrityError("SAM source must be the repository root")
    commit = _git(root, "rev-parse", "HEAD")
    if commit != FROZEN_SAM_COMMIT:
        raise SAMLoaderIntegrityError("SAM source commit mismatch")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise SAMLoaderIntegrityError("SAM source worktree is not clean")
    # Ignored Python source can shadow tracked imports despite a clean status.
    tracked = set(_git(root, "ls-files").splitlines())
    for package in ("segment_anything", "utils"):
        for path in (root / package).rglob("*.py"):
            if path.is_symlink() or path.relative_to(root).as_posix() not in tracked:
                raise SAMLoaderIntegrityError("Unverified Python source in SAM package")
    return commit


def _import_registry(root: Path) -> Mapping:
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in {"segment_anything", "utils"}:
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root):
                raise SAMLoaderIntegrityError(f"Conflicting imported module: {name}")
    previous_path = sys.path[:]
    try:
        sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        module = importlib.import_module("segment_anything")
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise SAMLoaderIntegrityError("SAM import came from another repository")
        registry = getattr(module, "sam_model_registry3D", None)
        if not isinstance(registry, Mapping) or not callable(registry.get(SAM_REGISTRY_KEY)):
            raise SAMLoaderIntegrityError("SAM registry requires callable vit_b_ori")
        return registry
    finally:
        sys.path[:] = previous_path


def load_sam_checkpoint(
    source_root: str | Path, checkpoint_path: str | Path,
) -> SAMCheckpointBundle:
    """Verify frozen provenance, strictly load on CPU, return eval/frozen model.

    The trusted frozen checkpoint requires weights_only=False. Its hash is
    checked before any external import or deserialization. No forward call is
    made. No fallback architecture, partial load, or device transfer is allowed.
    """
    root = Path(source_root).resolve(strict=True)
    path = Path(checkpoint_path).resolve(strict=True)
    commit = _verify_source(root)
    # Hash and deserialize the same open file, avoiding path replacement.
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != FROZEN_TURBO_SHA256:
            raise SAMLoaderIntegrityError("SAM turbo checkpoint SHA-256 mismatch")
        registry = _import_registry(root)
        with torch.device("cpu"):
            model = registry[SAM_REGISTRY_KEY](checkpoint=None)
        if not isinstance(model, torch.nn.Module):
            raise SAMLoaderIntegrityError("SAM constructor did not return a Module")
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("model_state_dict"), Mapping
    ):
        raise SAMLoaderIntegrityError("Checkpoint requires model_state_dict mapping")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise SAMLoaderIntegrityError("SAM model is not eval/frozen")
    if any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):
        raise SAMLoaderIntegrityError("SAM model must remain on CPU")
    return SAMCheckpointBundle(model, root, commit, path, digest)
