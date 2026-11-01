"""Operator-managed cortical asset registry; no patient data or mesh payloads."""

from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CortexMesh(RegistryModel):
    path: str
    sha256: SHA256

    @model_validator(mode="after")
    def relative_ply_path(self) -> Self:
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.path
            or path.suffix != ".ply"
        ):
            raise ValueError("Meshes require relative PLY paths without traversal.")
        return self


class CortexQC(RegistryModel):
    status: Literal["provisional", "approved", "rejected"]
    scope: Literal["research"]
    clinically_validated: Literal[False]
    notes: str = Field(min_length=1)


class CortexReconstruction(RegistryModel):
    reconstruction_id: str = Field(
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$"
    )
    source_t1n_sha256: SHA256
    asset_root: str = Field(min_length=1)
    coordinate_convention: Literal["physical RAS+"]
    units: Literal["mm"]
    qc: CortexQC
    meshes: dict[Literal["lh", "rh"], CortexMesh]

    @model_validator(mode="after")
    def complete_reconstruction(self) -> Self:
        root = PurePosixPath(self.asset_root)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("Asset storage requires an absolute directory.")
        if set(self.meshes) != {"lh", "rh"}:
            raise ValueError("Both cortical hemispheres must be registered.")
        if PurePosixPath(self.meshes["lh"].path) == PurePosixPath(
            self.meshes["rh"].path
        ):
            raise ValueError("Hemispheres require separate mesh files.")
        return self


class CortexRegistry(RegistryModel):
    schema_version: Literal[1]
    enabled: bool
    entries: list[CortexReconstruction]

    @model_validator(mode="after")
    def unambiguous_identities(self) -> Self:
        for field in ("reconstruction_id", "source_t1n_sha256"):
            values = [getattr(entry, field) for entry in self.entries]
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate registry {field}.")
        return self
