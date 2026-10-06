from typing import Literal

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    code: str
    message: str
    field: str | None = None


class ModalityValidation(BaseModel):
    field: str
    display_name: str
    research_core_name: str
    filename: str | None = None
    valid: bool | None = None


class GeometryInfo(BaseModel):
    shape_xyz: list[int]
    tensor_shape_dhw: list[int]
    voxel_spacing_xyz: list[float]
    affine: list[list[float]]
    affine_tolerance: float = 1e-5


class GroundTruthValidation(BaseModel):
    provided: bool
    filename: str | None = None
    valid: bool | None = None
    tensor_shape_dhw: list[int] | None = None
    labels: list[int] | None = None


class CaseValidationResponse(BaseModel):
    valid: bool
    status: Literal["ready", "invalid"]
    case_id: str | None = None

    modalities: dict[str, ModalityValidation]

    geometry: GeometryInfo | None = None
    ground_truth: GroundTruthValidation

    ready_for_inference: bool

    errors: list[ValidationIssue] = Field(default_factory=list)


class CaseDeleteResponse(BaseModel):
    case_id: str
    deleted: bool
