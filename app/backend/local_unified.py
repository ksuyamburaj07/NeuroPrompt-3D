"""Loopback-only, single-origin NeuroPrompt-3D application.

LOCAL DEVELOPMENT / REVIEW ONLY. Not an internet deployment.
Retains the existing FastAPI API, including writable routes.
"""

import hashlib
import os
import importlib.util
import re
from pathlib import Path
from threading import Lock
from types import ModuleType

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.backend.main import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIST = PROJECT_ROOT / "app/frontend/dist"

QC_ROOT = Path(
    os.environ.get(
        "NP3D_QC_RECONSTRUCTION_ROOT",
        "/data/NeuroPrompt3D_Experiments/"
        "cortical_reconstruction/m11e6b5_trial01",
    )
)
MESH_ROOT = QC_ROOT / "qc_review/np3d-b7d-real-qc"
EXPORT_ROOT = QC_ROOT / "viewer_exports"
MRI_SCRIPT = (
    PROJECT_ROOT / "app/backend/services/pinned_cortical_qc.py"
)

ALLOWED_MESHES = frozenset({
    "lh.pial.ras.ply",
    "rh.pial.ras.ply",
    "lh.pial.ras.sulc-qc.ply",
    "rh.pial.ras.sulc-qc.ply",
})

ALLOWED_PLANES = frozenset({
    "axial", "coronal", "sagittal",
})
ALLOWED_KINDS = frozenset({
    "slices", "slices3d", "contours",
})

_qc_lock = Lock()
_qc_module: ModuleType | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def qc_renderer() -> ModuleType:
    """Lazy, one-time import of the original verified renderer."""
    global _qc_module

    if _qc_module is not None:
        return _qc_module

    with _qc_lock:
        if _qc_module is None:
            if not MRI_SCRIPT.is_file() or MRI_SCRIPT.is_symlink():
                raise RuntimeError("Pinned MRI QC renderer unavailable.")

            spec = importlib.util.spec_from_file_location(
                "np3d_pinned_local_mri_qc",
                MRI_SCRIPT,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Unable to load MRI QC renderer.")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # The original script's initialize() checks:
            # - original T1N MRI SHA-256
            # - canonical RAS+ orientation
            # - original pial meshes against verified exports
            _qc_module = module

    return _qc_module


def reject_query(request: Request) -> None:
    if request.scope.get("query_string", b""):
        raise HTTPException(
            status_code=400,
            detail="Unexpected query parameters.",
        )


if not (DIST / "index.html").is_file():
    raise RuntimeError(
        "Production frontend missing. Run npm run build first."
    )

if not (DIST / "assets").is_dir():
    raise RuntimeError("Production assets directory missing.")


app = create_app()


@app.middleware("http")
async def local_only(request: Request, call_next):
    """Reject remote clients and foreign browser origins."""
    client = request.client.host if request.client else ""
    host = request.headers.get("host", "")
    origin = request.headers.get("origin")

    valid_host = re.fullmatch(
        r"(?:127\.0\.0\.1|localhost):[0-9]{1,5}",
        host,
    ) is not None

    if client != "127.0.0.1" or not valid_host:
        return Response(
            status_code=403,
            content="Local access only.",
        )

    if origin is not None and origin != f"http://{host}":
        return Response(
            status_code=403,
            content="Foreign origin rejected.",
        )

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/qc/meshes/{filename}", include_in_schema=False)
def cortical_mesh(filename: str, request: Request):
    reject_query(request)

    if filename not in ALLOWED_MESHES:
        raise HTTPException(status_code=404)

    mesh = MESH_ROOT / filename

    if (
        MESH_ROOT.is_symlink()
        or not mesh.is_file()
        or mesh.is_symlink()
    ):
        raise HTTPException(
            status_code=503,
            detail="Approved cortical mesh unavailable.",
        )

    # Original pial meshes must still match their exports.
    if filename in {
        "lh.pial.ras.ply",
        "rh.pial.ras.ply",
    }:
        exported = EXPORT_ROOT / filename
        if (
            not exported.is_file()
            or exported.is_symlink()
            or file_sha256(mesh) != file_sha256(exported)
        ):
            raise HTTPException(
                status_code=503,
                detail="Cortical mesh verification failed.",
            )

    return FileResponse(
        mesh,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/qc/mri/metadata", include_in_schema=False)
def cortical_metadata(request: Request):
    reject_query(request)

    try:
        renderer = qc_renderer()
        payload = renderer.metadata()
    except Exception as error:
        print(
            "Local MRI metadata unavailable:",
            type(error).__name__,
            flush=True,
        )
        raise HTTPException(
            status_code=503,
            detail="Verified MRI QC metadata unavailable.",
        ) from None

    return JSONResponse(
        payload,
        headers={"Cache-Control": "no-store"},
    )


@app.get(
    "/qc/mri/{kind}/{plane}/{index}",
    include_in_schema=False,
)
def cortical_png(
    kind: str,
    plane: str,
    index: str,
    request: Request,
):
    reject_query(request)

    if (
        kind not in ALLOWED_KINDS
        or plane not in ALLOWED_PLANES
    ):
        raise HTTPException(status_code=404)

    if re.fullmatch(r"[0-9]{1,3}", index) is None:
        raise HTTPException(status_code=400)

    try:
        renderer = qc_renderer()
    except Exception as error:
        print(
            "Local MRI renderer unavailable:",
            type(error).__name__,
            flush=True,
        )
        raise HTTPException(
            status_code=503,
            detail="Verified MRI QC renderer unavailable.",
        ) from None

    position = int(index)

    if position >= renderer.SHAPE[renderer.PLANES[plane]]:
        raise HTTPException(
            status_code=400,
            detail="Slice index out of range.",
        )

    try:
        payload = renderer.render_png(
            kind, plane, position
        )
    except Exception as error:
        print(
            "Local MRI rendering failed:",
            type(error).__name__,
            flush=True,
        )
        raise HTTPException(
            status_code=503,
            detail="MRI QC rendering unavailable.",
        ) from None

    return Response(
        content=payload,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# Only the generated frontend asset directory is mounted.
app.mount(
    "/assets",
    StaticFiles(directory=str(DIST / "assets")),
    name="unified-frontend-assets",
)


@app.get("/", include_in_schema=False)
def frontend_home():
    return FileResponse(
        DIST / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/{path:path}", include_in_schema=False)
def frontend_navigation(path: str):
    """SPA fallback; never reinterpret missing API routes as HTML."""
    if (
        path.split("/", 1)[0] in {
            "api", "qc", "assets", "docs", "redoc"
        }
        or "." in path
    ):
        raise HTTPException(status_code=404)

    return FileResponse(
        DIST / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )
