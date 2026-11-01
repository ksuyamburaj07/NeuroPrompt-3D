"""Restricted discovery of persisted development runs.

Not production authentication. Disabled by default.
Do not expose this endpoint through a public reverse proxy.
"""

from __future__ import annotations

import os
from datetime import datetime

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    Response,
)
from pydantic import BaseModel, ConfigDict

from app.backend.services.research_discovery_access import (
    local_research_discovery_allowed,
)
from app.backend.services.research_run_listing_service import (
    list_recent_live_runs,
)


router = APIRouter(
    prefix="/research",
    tags=["research-discovery"],
)

NO_STORE_HEADERS = {
    "Cache-Control": "private, no-store",
    "X-Content-Type-Options": "nosniff",
}


class RecentResearchRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    stage: str
    progress: float
    updated_at: datetime


class RecentResearchRunsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs: list[RecentResearchRun]
    count: int
    limit: int


@router.get(
    "/runs",
    response_model=RecentResearchRunsResponse,
)
def get_recent_research_runs(
    request: Request,
    response: Response,
    limit: int = Query(default=20, ge=1, le=50),
) -> RecentResearchRunsResponse:
    """Read-only local development discovery.

    This guard is not a replacement for authentication,
    authorization or deployment network isolation.
    """
    client_host = (
        request.client.host
        if request.client is not None
        else None
    )

    permitted = local_research_discovery_allowed(
        enabled=os.environ.get(
            "NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY"
        ),
        client_host=client_host,
        request_host=request.headers.get("host"),
        origin=request.headers.get("origin"),
    )

    if not permitted:
        raise HTTPException(
            status_code=404,
            detail="Not found",
            headers=NO_STORE_HEADERS,
        )

    try:
        records = list_recent_live_runs(limit=limit)
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Research run inventory is unavailable.",
            headers=NO_STORE_HEADERS,
        ) from None

    for name, value in NO_STORE_HEADERS.items():
        response.headers[name] = value

    response.headers["Vary"] = "Origin"

    summaries = [
        RecentResearchRun(
            run_id=record.run_id,
            status=record.status,
            stage=record.stage,
            progress=record.progress,
            updated_at=record.updated_at,
        )
        for record in records
    ]

    return RecentResearchRunsResponse(
        runs=summaries,
        count=len(summaries),
        limit=limit,
    )
