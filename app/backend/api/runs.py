from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    status,
)
from fastapi.responses import FileResponse

from app.backend.schemas.runs import (
    RunView,
)
from app.backend.services.artifact_service import (
    ArtifactIntegrityError,
    ArtifactNotFound,
    ArtifactRunNotComplete,
    resolve_live_artifact,
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
    response_model=RunView,
    status_code=status.HTTP_201_CREATED,
)
def create_run(
    case_id: str,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
    ),
) -> RunView:
    try:
        return RunView.from_record(
            create_live_run(
                case_id,
                idempotency_key=idempotency_key,
            )
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
    response_model=RunView,
)
def get_run(
    run_id: str,
) -> RunView:
    try:
        return RunView.from_record(
            load_live_run(
                run_id
            )
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
    response_model=RunView,
    status_code=status.HTTP_202_ACCEPTED,
)
def execute_run(
    run_id: str,
) -> RunView:
    try:
        return RunView.from_record(
            launch_run_worker(
                run_id
            )
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


@router.get(
    "/runs/{run_id}/artifacts/{artifact_name}",
    response_class=FileResponse,
)
def get_run_artifact(
    run_id: str,
    artifact_name: str,
) -> FileResponse:
    try:
        artifact = resolve_live_artifact(
            run_id,
            artifact_name,
        )

    except ArtifactRunNotComplete as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except ArtifactNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail="Live artifact not found",
        ) from exc

    except ArtifactIntegrityError as exc:
        raise HTTPException(
            status_code=500,
            detail="Live artifact integrity check failed",
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

    return FileResponse(
        path=artifact.path,
        media_type=artifact.media_type,
        filename=artifact.filename,
    )
