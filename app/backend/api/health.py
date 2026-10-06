from fastapi import APIRouter

from app.backend.core.constants import (
    APP_NAME,
    APP_VERSION,
    FROZEN_RESEARCH_CORE_COMMIT,
    PRE_M11_PROVENANCE_COMMIT,
    RESEARCH_DISCLAIMER,
)
from app.backend.schemas.health import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        application=APP_NAME,
        application_version=APP_VERSION,
        frozen_research_core_commit=FROZEN_RESEARCH_CORE_COMMIT,
        pre_m11_provenance_commit=PRE_M11_PROVENANCE_COMMIT,
        m9e_read_only=True,
        disclaimer=RESEARCH_DISCLAIMER,
    )
