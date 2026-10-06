from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.backend.schemas.runs import (
    RunRecord,
)
from app.backend.services.run_service import (
    create_live_run,
    load_live_run,
)
from app.backend.services.worker_service import (
    RunLaunchConflict,
    launch_run_worker,
)


router = APIRouter(
    tags=["live inference runs"]
)


@router.post(
    "/cases/{case_id}/runs",
    response_model=RunRecord,
    status_code=status.HTTP_201_CREATED,
)
def create_run(
    case_id: str,
) -> RunRecord:
    try:
        return create_live_run(
            case_id
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Live case not found",
        ) from exc


@router.get(
    "/runs/{run_id}",
    response_model=RunRecord,
)
def get_run(
    run_id: str,
) -> RunRecord:
    try:
        return load_live_run(
            run_id
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Live inference run not found",
        ) from exc

@router.post(
    "/runs/{run_id}/execute",
    response_model=RunRecord,
    status_code=status.HTTP_202_ACCEPTED,
)
def execute_run(
    run_id: str,
) -> RunRecord:
    try:
        return launch_run_worker(
            run_id
        )

    except RunLaunchConflict as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Live inference run not found",
        ) from exc

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to launch inference worker",
        ) from exc
