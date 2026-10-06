from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    application: str
    application_version: str
    frozen_research_core_commit: str
    pre_m11_provenance_commit: str
    m9e_read_only: bool
    disclaimer: str
