from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


RunStatus = Literal[
    "queued",
    "running",
    "complete",
    "failed",
]

RunStage = Literal[
    "queued",
    "validating",
    "preprocessing",
    "baseline",
    "mc_dropout",
    "hotspot",
    "policy",
    "sam_refinement",
    "finalizing",
    "complete",
    "failed",
]


class RunError(BaseModel):
    code: str
    message: str


class RunRecord(BaseModel):
    run_id: str
    case_id: str

    status: RunStatus
    stage: RunStage

    progress: float = Field(
        ge=0.0,
        le=1.0,
    )

    created_at: datetime
    updated_at: datetime

    frozen_variance_threshold: float

    action: str | None = None
    gate_state: str | None = None

    hotspot_zyx: list[int] | None = None
    hotspot_variance: float | None = None

    fp_prompt_zyx: list[int] | None = None
    fp_prompt_model_xyz: list[int] | None = None

    sam_used: bool | None = None
    sam_refinement_skipped: bool | None = None

    semantic_abstention_condition: str | None = None

    artifacts: dict[str, str] = Field(
        default_factory=dict
    )

    error: RunError | None = None
