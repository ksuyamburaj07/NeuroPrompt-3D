from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.backend.schemas.cases import (
    CaseDeleteResponse,
    CaseValidationResponse,
)
from app.backend.services.case_service import (
    delete_live_case,
    load_live_case,
    validate_and_stage_case,
)


router = APIRouter(tags=["live cases"])


@router.post(
    "/cases/validate",
    response_model=CaseValidationResponse,
)
async def validate_case(
    t1n: UploadFile | None = File(default=None),
    t1c: UploadFile | None = File(default=None),
    t2w: UploadFile | None = File(default=None),
    t2f: UploadFile | None = File(default=None),
    segmentation: UploadFile | None = File(default=None),
) -> CaseValidationResponse:
    return await validate_and_stage_case(
        uploads={
            "t1n": t1n,
            "t1c": t1c,
            "t2w": t2w,
            "t2f": t2f,
        },
        segmentation=segmentation,
    )


@router.get(
    "/cases/{case_id}",
    response_model=CaseValidationResponse,
)
def get_case(
    case_id: str,
) -> CaseValidationResponse:
    try:
        return load_live_case(case_id)
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


@router.delete(
    "/cases/{case_id}",
    response_model=CaseDeleteResponse,
)
def delete_case(
    case_id: str,
) -> CaseDeleteResponse:
    try:
        deleted = delete_live_case(case_id)
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

    return CaseDeleteResponse(
        case_id=case_id,
        deleted=deleted,
    )
