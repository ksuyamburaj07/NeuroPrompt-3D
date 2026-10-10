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
from app.backend.services.run_service import (
    active_live_run_ids_for_case,
)


from fastapi.responses import Response

from app.backend.services.viewer_service import (
    viewer_metadata,
    viewer_slice_png,
)



from fastapi.responses import Response

from app.backend.services.artifact_service import (
    ArtifactIntegrityError,
    ArtifactNotFound,
    ArtifactRunNotComplete,
)
from app.backend.services.viewer_overlay_service import (
    OverlayCaseMismatch,
    OverlayGeometryMismatch,
    overlay_slice_png,
)


from app.backend.services.viewer_marker_service import (
    ViewerCoordinateError,
    ViewerRunCaseMismatch,
    viewer_run_markers,
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
        active_run_ids = (
            active_live_run_ids_for_case(
                case_id
            )
        )

        if active_run_ids:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Live case cannot be deleted "
                    "while an inference run is "
                    "queued or running."
                ),
            )

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


@router.get("/cases/{case_id}/viewer/metadata")
def get_case_viewer_metadata(case_id: str) -> dict:
    try:
        return viewer_metadata(case_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Staged MRI case or modality not found.",
        ) from exc


@router.get("/cases/{case_id}/viewer/slices/{modality}/{plane}/{index}")
def get_case_viewer_slice(
    case_id: str,
    modality: str,
    plane: str,
    index: int,
) -> Response:
    try:
        image_bytes = viewer_slice_png(
            case_id, modality, plane, index
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Staged MRI case or modality not found.",
        ) from exc

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@router.get(
    "/cases/{case_id}/viewer/overlays/{run_id}/{layer}/{plane}/{index}"
)
def get_case_viewer_overlay(
    case_id: str,
    run_id: str,
    layer: str,
    plane: str,
    index: int,
) -> Response:
    try:
        data = overlay_slice_png(
            case_id, run_id, layer, plane, index
        )
    except (OverlayCaseMismatch, OverlayGeometryMismatch) as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
    except ArtifactRunNotComplete as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc
    except ArtifactNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail="Segmentation artifact unavailable.",
        ) from exc
    except ArtifactIntegrityError as exc:
        raise HTTPException(
            status_code=500,
            detail="Segmentation artifact integrity check failed.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="MRI case or inference run unavailable.",
        ) from exc

    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/cases/{case_id}/viewer/markers/{run_id}")
def get_case_viewer_markers(case_id: str, run_id: str) -> dict:
    try:
        return viewer_run_markers(case_id, run_id)

    except (ViewerCoordinateError, ViewerRunCaseMismatch) as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except ArtifactRunNotComplete as exc:
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
            detail="MRI case or inference run unavailable.",
        ) from exc


@router.get("/cases/{case_id}/viewer/meshes/{run_id}/{layer}")
def get_case_viewer_mesh(
    case_id: str,
    run_id: str,
    layer: str,
) -> Response:
    from app.backend.services.viewer_mesh_service import (
        ViewerMeshCaseMismatch,
        ViewerMeshGeometryError,
        viewer_mesh_ply,
    )

    try:
        mesh_bytes = viewer_mesh_ply(case_id, run_id, layer)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except ArtifactRunNotComplete as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    except (ViewerMeshCaseMismatch, ViewerMeshGeometryError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="MRI case, run or mask artifact unavailable.",
        ) from exc

    return Response(
        content=mesh_bytes,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Mesh-Coordinates": "RAS+ millimetres",
        },
    )


@router.get("/cases/{case_id}/viewer/anatomy/brain")
def get_case_viewer_anatomy(
    case_id: str,
    step: int = 2,
    finish: str = "raw",
) -> Response:
    from app.backend.services.viewer_anatomy_service import (
        ANATOMY_STEP_VOXELS,
        viewer_anatomy_ply,
    )
    from app.backend.services.viewer_mesh_service import (
        ViewerMeshGeometryError,
    )

    try:
        mesh_bytes = viewer_anatomy_ply(
            case_id, step=step, finish=finish
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ViewerMeshGeometryError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Staged anatomical MRI unavailable.",
        ) from exc

    return Response(
        content=mesh_bytes,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Mesh-Coordinates": "RAS+ millimetres",
            "X-Mesh-Source": "t1n-nonzero-foreground",
            "X-Mesh-Step-Voxels": str(step),
            "X-Mesh-Finish": finish,
        },
    )
