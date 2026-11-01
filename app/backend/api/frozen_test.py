"""Read-only API for verified M9E final-test evidence."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Path, Query, Response, status

from app.backend.services.frozen_test_service import (
    FrozenEvidenceError,
    get_frozen_test_case,
    get_frozen_test_cases,
    get_frozen_test_summary,
)

from app.backend.services.frozen_slice_service import (
    FrozenSliceError,
    render_frozen_axial_png,
)

from app.backend.services.frozen_initial_position_service import get_verified_initial_position

from app.backend.services.frozen_removed_focus_service import get_verified_removed_focus

router = APIRouter(tags=["frozen test evidence"])


@router.get("/frozen-test/summary")
def frozen_test_summary(response: Response) -> dict:
    try:
        summary = get_frozen_test_summary()
    except FrozenEvidenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verified frozen test evidence is unavailable.",
        ) from exc

    response.headers["Cache-Control"] = "no-store"
    return summary


@router.get("/frozen-test/cases")
def frozen_test_cases(
    response: Response,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=129),
    action: str | None = Query(default=None),
) -> dict:
    """Read-only, paginated list of verified frozen cases."""

    try:
        page = get_frozen_test_cases(
            offset=offset,
            limit=limit,
            action=action,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid frozen case-list parameters.",
            headers={"Cache-Control": "no-store"},
        ) from exc
    except FrozenEvidenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verified frozen test evidence is unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from exc

    response.headers["Cache-Control"] = "no-store"

    return page


@router.get("/frozen-test/cases/{case_index}")
def frozen_test_case(
    response: Response,
    case_index: int = Path(ge=1, le=129),
) -> dict:
    """Read-only verified measurements for one case."""

    try:
        result = get_frozen_test_case(
            case_index=case_index,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid frozen case index.",
            headers={"Cache-Control": "no-store"},
        ) from exc
    except FrozenEvidenceError as exc:
        raise HTTPException(
            status_code=503,
            detail="Verified frozen test evidence is unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from exc

    response.headers["Cache-Control"] = "no-store"
    return result


@router.get("/frozen-test/cases/{case_index}/slices/axial")
def frozen_test_axial_slice(
    case_index: int = Path(ge=1, le=129),
    z: int = Query(ge=0, le=181),
    modality: Literal["t1n", "t1c", "t2w", "t2f"] = Query(
        default="t2f"
    ),
    layer: Literal[
        "none", "baseline", "final", "removed", "support"
    ] = Query(default="removed"),
    contrast: Literal["slice", "volume"] = Query(default="slice"),
) -> Response:
    """Return an authenticated, read-only frozen axial preview."""

    try:
        render_kwargs = {
            "modality": modality,
            "layer": layer,
        }

        # Preserve the accepted legacy renderer call.
        # Explicit volume contrast uses the new parameter.
        if contrast == "volume":
            render_kwargs["contrast"] = contrast

        png = render_frozen_axial_png(
            case_index,
            z,
            **render_kwargs,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid frozen slice parameters.",
            headers={"Cache-Control": "no-store"},
        ) from exc

    except FrozenSliceError as exc:
        raise HTTPException(
            status_code=503,
            detail="Verified frozen MRI evidence is unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _nonaxial_frozen_png_response(
    case_index: int,
    plane: str,
    index: int,
    modality: str,
    layer: str,
    contrast: str,
):
    """Return only authenticated frozen MRI evidence."""

    from fastapi import HTTPException
    from fastapi.responses import Response

    from app.backend.services.frozen_multiplane_service import (
        render_frozen_nonaxial_png,
    )

    try:
        png = render_frozen_nonaxial_png(
            case_index,
            plane,
            index,
            modality=modality,
            layer=layer,
            contrast=contrast,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid frozen MRI slice request.",
        ) from exc
    except Exception as exc:
        # Never expose local paths, case identities, or
        # internal evidence-integrity errors to clients.
        raise HTTPException(
            status_code=503,
            detail="Verified frozen MRI slice unavailable.",
        ) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/frozen-test/cases/{case_index}/slices/coronal")
def get_frozen_coronal_slice(
    case_index: int,
    index: int,
    modality: str = "t2f",
    layer: str = "removed",
    contrast: Literal["slice", "volume"] = Query(default="slice"),
):
    return _nonaxial_frozen_png_response(
        case_index, "coronal", index, modality, layer, contrast
    )


@router.get("/frozen-test/cases/{case_index}/slices/sagittal")
def get_frozen_sagittal_slice(
    case_index: int,
    index: int,
    modality: str = "t2f",
    layer: str = "removed",
    contrast: Literal["slice", "volume"] = Query(default="slice"),
):
    return _nonaxial_frozen_png_response(
        case_index, "sagittal", index, modality, layer, contrast
    )


@router.get("/frozen-test/cases/{case_index}/initial-position")
def frozen_test_initial_position(
    response: Response,
    case_index: int = Path(ge=1, le=129),
) -> dict:
    """Return a verified, deterministic native XYZ display position."""
    try:
        result = get_verified_initial_position(case_index)
    except (FrozenEvidenceError, FrozenSliceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verified frozen positioning evidence is unavailable.",
        ) from exc

    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return result


@router.get("/frozen-test/cases/{case_index}/removed-focus")
def frozen_test_removed_focus(
    response: Response,
    case_index: int = Path(ge=1, le=129),
) -> dict:
    """Return an authenticated, read-only refinement focus."""
    try:
        result = get_verified_removed_focus(case_index)
    except (FrozenEvidenceError, FrozenSliceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verified frozen removal evidence is unavailable.",
            headers={"Cache-Control": "no-store"},
        ) from exc

    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return result
