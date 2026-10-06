"""Safe logical-name delivery for completed live-run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.backend.core import paths
from app.backend.services.run_service import (
    load_live_run,
)


class ArtifactRunNotComplete(RuntimeError):
    """Artifacts are unavailable until the run reaches COMPLETE."""


class ArtifactNotFound(FileNotFoundError):
    """Requested logical artifact is unavailable."""


class ArtifactIntegrityError(RuntimeError):
    """Persisted artifact metadata violates the application contract."""


@dataclass(frozen=True)
class ArtifactSpec:
    filename: str
    media_type: str


@dataclass(frozen=True)
class ResolvedArtifact:
    path: Path
    filename: str
    media_type: str


ARTIFACT_SPECS: dict[
    str,
    ArtifactSpec,
] = {
    "baseline_probability_npy": ArtifactSpec(
        filename="baseline_probability.npy",
        media_type="application/octet-stream",
    ),
    "baseline_probability_nifti": ArtifactSpec(
        filename="baseline_probability.nii.gz",
        media_type="application/gzip",
    ),
    "baseline_mask_npy": ArtifactSpec(
        filename="baseline_mask.npy",
        media_type="application/octet-stream",
    ),
    "baseline_mask_nifti": ArtifactSpec(
        filename="baseline_mask.nii.gz",
        media_type="application/gzip",
    ),
    "final_mask_npy": ArtifactSpec(
        filename="final_mask.npy",
        media_type="application/octet-stream",
    ),
    "final_mask_nifti": ArtifactSpec(
        filename="final_mask.nii.gz",
        media_type="application/gzip",
    ),
    "removal_mask_npy": ArtifactSpec(
        filename="removal_mask.npy",
        media_type="application/octet-stream",
    ),
    "removal_mask_nifti": ArtifactSpec(
        filename="removal_mask.nii.gz",
        media_type="application/gzip",
    ),
    "support_mask_npy": ArtifactSpec(
        filename="support_mask.npy",
        media_type="application/octet-stream",
    ),
    "support_mask_nifti": ArtifactSpec(
        filename="support_mask.nii.gz",
        media_type="application/gzip",
    ),
    "result_json": ArtifactSpec(
        filename="result.json",
        media_type="application/json",
    ),
}


def resolve_live_artifact(
    run_id: str,
    artifact_name: str,
) -> ResolvedArtifact:
    """Resolve one approved artifact without accepting filesystem paths."""

    run = load_live_run(
        run_id
    )

    if run.status != "complete":
        raise ArtifactRunNotComplete(
            "Live artifacts are available only after run completion."
        )

    spec = ARTIFACT_SPECS.get(
        artifact_name
    )

    if spec is None:
        raise ArtifactNotFound(
            artifact_name
        )

    declared = run.artifacts.get(
        artifact_name
    )

    if declared is None:
        raise ArtifactNotFound(
            artifact_name
        )

    expected_relative = (
        Path("artifacts")
        / spec.filename
    )

    declared_relative = Path(
        declared
    )

    # The run record must contain exactly the path produced by the
    # application artifact writer. Arbitrary relative or absolute paths
    # are never accepted.
    if (
        declared_relative.is_absolute()
        or declared_relative
        != expected_relative
    ):
        raise ArtifactIntegrityError(
            "Persisted artifact path violates the approved artifact contract."
        )

    runtime_root = (
        paths.LIVE_RUNS_ROOT
        .resolve(
            strict=False
        )
    )

    run_root = (
        paths.LIVE_RUNS_ROOT
        / run_id
    )

    if run_root.is_symlink():
        raise ArtifactIntegrityError(
            "Live run directory must not be a symbolic link."
        )

    resolved_run_root = (
        run_root.resolve(
            strict=False
        )
    )

    try:
        resolved_run_root.relative_to(
            runtime_root
        )

    except ValueError as exc:
        raise ArtifactIntegrityError(
            "Live run resolved outside the approved runtime directory."
        ) from exc

    artifact_root_path = (
        run_root
        / "artifacts"
    )

    if artifact_root_path.is_symlink():
        raise ArtifactIntegrityError(
            "Live artifact directory must not be a symbolic link."
        )

    artifact_root = (
        artifact_root_path.resolve(
            strict=False
        )
    )

    try:
        artifact_root.relative_to(
            resolved_run_root
        )

    except ValueError as exc:
        raise ArtifactIntegrityError(
            "Live artifact directory resolved outside its run directory."
        ) from exc

    stored_path = (
        run_root
        / declared_relative
    )

    # A symlink could otherwise point a syntactically valid artifact name
    # outside the live-run directory.
    if stored_path.is_symlink():
        raise ArtifactIntegrityError(
            "Live artifact must not be a symbolic link."
        )

    resolved = stored_path.resolve(
        strict=False
    )

    try:
        resolved.relative_to(
            artifact_root
        )

    except ValueError as exc:
        raise ArtifactIntegrityError(
            "Live artifact resolved outside its approved artifact directory."
        ) from exc

    if not resolved.is_file():
        raise ArtifactNotFound(
            artifact_name
        )

    return ResolvedArtifact(
        path=resolved,
        filename=spec.filename,
        media_type=spec.media_type,
    )
