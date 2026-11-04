"""Protected read-only viewer for real NeuroPrompt-3D live runs.

Reuses original M11 viewer, overlay, marker and mesh services.
Never creates or executes an inference run.
"""

import modal

app = modal.App("neuroprompt3d-live-viewer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.14.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "fastapi==0.142.2",
        "numpy==2.4.6",
        "nibabel==5.4.2",
        "scipy==1.17.1",
        "Pillow==12.3.0",
        "scikit-image==0.26.0",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir("app/backend/core", "/root/app/backend/core")
    .add_local_dir("app/backend/schemas", "/root/app/backend/schemas")
    .add_local_dir("app/backend/services", "/root/app/backend/services")
)

runtime = modal.Volume.from_name("np3d-live-runtime")

MOUNT = {
    "/root/app/backend/runtime":
        runtime.with_mount_options(read_only=True)
}

CASE_ID = "case_d5c817d9931a4c1ba3f09426c43006fc"
RUN_ID = "run_aae795bb5d754c70ad20789724c71cc7"


@app.function(
    image=image,
    cpu=1,
    memory=4096,
    timeout=240,
    max_containers=1,
    volumes=MOUNT,
)
def verify_viewer():
    import gzip
    from app.backend.services.viewer_service import (
        viewer_metadata, viewer_slice_png,
    )
    from app.backend.services.viewer_overlay_service import (
        overlay_slice_png,
    )
    from app.backend.services.viewer_marker_service import (
        viewer_run_markers,
    )
    from app.backend.services.viewer_mesh_service import (
        viewer_mesh_ply,
    )

    meta = viewer_metadata(CASE_ID)
    assert meta["case_id"] == CASE_ID

    mri = viewer_slice_png(CASE_ID, "t2f", "axial", 96)
    overlay = overlay_slice_png(
        CASE_ID, RUN_ID, "final", "axial", 96
    )

    signature = b"\x89PNG\r\n\x1a\n"
    assert mri.startswith(signature)
    assert overlay.startswith(signature)

    markers = viewer_run_markers(CASE_ID, RUN_ID)
    assert markers["run_id"] == RUN_ID
    assert markers["case_id"] == CASE_ID

    mesh = viewer_mesh_ply(CASE_ID, RUN_ID, "final")
    assert mesh.startswith(b"ply\n")

    return {
        "case_id": CASE_ID,
        "run_id": RUN_ID,
        "shape_xyz": meta["shape_ras_xyz"],
        "mri_png_bytes": len(mri),
        "overlay_png_bytes": len(overlay),
        "mesh_bytes": len(mesh),
        "mesh_gzip_bytes": len(gzip.compress(mesh, compresslevel=6)),
        "markers_verified": True,
    }



@app.function(
    image=image,
    cpu=1,
    memory=4096,
    max_containers=1,
    volumes=MOUNT,
)
def verify_new_case():
    from app.backend.services.viewer_service import (
        viewer_metadata,
        viewer_slice_png,
    )

    case_id = "case_6d362b430e614eb1a483f553aa234ba0"

    metadata = viewer_metadata(case_id)

    assert metadata["case_id"] == case_id
    assert metadata["shape_ras_xyz"] == [182, 218, 182]

    png = viewer_slice_png(
        case_id, "t1n", "axial", 91
    )

    assert png.startswith(bytes.fromhex("89504e470d0a1a0a"))

    return {
        "case_id": case_id,
        "shape": metadata["shape_ras_xyz"],
        "slice_bytes": len(png),
        "success": True,
    }


@app.local_entrypoint()
def verify_uploaded_case():
    result = verify_new_case.remote()
    assert result["success"]
    print("PASS: Uploaded browser case visible in Modal")
    print("PASS: Original MRI slice renderer")
    print(result)


@app.local_entrypoint()
def verify():
    result = verify_viewer.remote()

    assert result["markers_verified"]
    print("PASS: Original live MRI metadata")
    print("PASS: Original live MRI PNG")
    print("PASS: Original saved prediction overlay")
    print("PASS: Original scientific markers")
    print("PASS: Original final segmentation mesh")
    print(result)


@app.function(
    image=image,
    cpu=1,
    memory=4096,
    timeout=240,
    max_containers=1,
    volumes=MOUNT,
)
@modal.asgi_app(requires_proxy_auth=True)
def live_viewer_api():
    import re

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response

    from app.backend.services.viewer_service import (
        viewer_metadata, viewer_slice_png,
    )
    from app.backend.services.viewer_overlay_service import (
        overlay_slice_png,
    )
    from app.backend.services.viewer_marker_service import (
        viewer_run_markers,
    )
    from app.backend.services.viewer_mesh_service import (
        viewer_mesh_ply,
    )
    from app.backend.services.viewer_anatomy_service import (
        viewer_anatomy_ply,
    )
    from app.backend.services.viewer_cortex_service import (
        viewer_cortex_status, viewer_cortex_ply,
    )
    from app.backend.services.viewer_cortex_contour_service import (
        viewer_cortex_contours_png,
    )

    api = FastAPI(
        title="NeuroPrompt-3D Protected Live Viewer",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Refresh cases committed by the separate upload container.
    import asyncio
    refresh_lock = asyncio.Lock()

    @api.middleware("http")
    async def refresh_runtime(request, call_next):
        async with refresh_lock:
            runtime.reload()
            return await call_next(request)

    case_pattern = re.compile(r"case_[0-9a-f]{32}")
    run_pattern = re.compile(r"run_[0-9a-f]{32}")

    planes = {"axial", "coronal", "sagittal"}
    modalities = {"t1n", "t1c", "t2w", "t2f"}
    layers = {"baseline", "final", "removed"}

    def valid_index(value):
        return (
            value.isascii()
            and value.isdecimal()
            and len(value) <= 3
            and int(value) <= 511
        )

    @api.get("/api/v1/cases/{case_id}/viewer/{tail:path}")
    def view(case_id: str, tail: str, request: Request):
        if not case_pattern.fullmatch(case_id):
            raise HTTPException(400, "Invalid case identifier")

        parts = tail.split("/")
        query = request.query_params

        if any(
            key not in {"transparent_background", "step", "finish"}
            for key in query.keys()
        ):
            raise HTTPException(400, "Unexpected query parameter")

        try:
            media_type = "application/json"

            if parts == ["metadata"]:
                if query:
                    raise ValueError("Metadata does not accept options")
                result = viewer_metadata(case_id)

            elif (
                len(parts) == 4
                and parts[0] == "slices"
                and parts[1] in modalities
                and parts[2] in planes
                and valid_index(parts[3])
            ):
                transparent = query.get(
                    "transparent_background", "false"
                )
                if transparent not in {"true", "false"}:
                    raise ValueError("Invalid transparency option")
                if any(k != "transparent_background" for k in query):
                    raise ValueError("Unexpected slice option")

                result = viewer_slice_png(
                    case_id, parts[1], parts[2], int(parts[3]),
                    transparent_background=(transparent == "true"),
                )
                media_type = "image/png"

            elif (
                len(parts) == 5
                and parts[0] == "overlays"
                and run_pattern.fullmatch(parts[1])
                and parts[2] in layers
                and parts[3] in planes
                and valid_index(parts[4])
            ):
                if query:
                    raise ValueError("Unexpected overlay option")
                result = overlay_slice_png(
                    case_id, parts[1], parts[2], parts[3],
                    int(parts[4]),
                )
                media_type = "image/png"

            elif (
                len(parts) == 2
                and parts[0] == "markers"
                and run_pattern.fullmatch(parts[1])
            ):
                if query:
                    raise ValueError("Unexpected marker option")
                result = viewer_run_markers(case_id, parts[1])

            elif (
                len(parts) == 3
                and parts[0] == "meshes"
                and run_pattern.fullmatch(parts[1])
                and parts[2] in layers
            ):
                if query:
                    raise ValueError("Unexpected mesh option")
                result = viewer_mesh_ply(
                    case_id, parts[1], parts[2]
                )
                media_type = "application/octet-stream"

            elif parts == ["anatomy", "brain"]:
                step = query.get("step", "2")
                finish = query.get("finish", "raw")
                if step not in {"1", "2"}:
                    raise ValueError("Invalid anatomy step")
                if finish not in {"raw", "soft"}:
                    raise ValueError("Invalid anatomy finish")
                result = viewer_anatomy_ply(
                    case_id, step=int(step), finish=finish
                )
                media_type = "application/octet-stream"

            elif parts == ["cortex", "status"]:
                if query:
                    raise ValueError("Unexpected cortex option")
                result = viewer_cortex_status(case_id)

            elif (
                len(parts) == 2
                and parts[0] == "cortex"
                and parts[1] in {"lh", "rh"}
            ):
                if query:
                    raise ValueError("Unexpected cortex option")
                result = viewer_cortex_ply(case_id, parts[1])
                media_type = "application/octet-stream"

            elif (
                len(parts) == 4
                and parts[0:2] == ["cortex", "contours"]
                and parts[2] in planes
                and valid_index(parts[3])
            ):
                if query:
                    raise ValueError("Unexpected cortex option")
                result = viewer_cortex_contours_png(
                    case_id, parts[2], int(parts[3])
                )
                media_type = "image/png"

            else:
                raise HTTPException(404, "Viewer route unavailable")

        except FileNotFoundError:
            raise HTTPException(
                404, "Verified case or artifact unavailable"
            ) from None
        except ValueError:
            raise HTTPException(
                400, "Invalid or incompatible viewer request"
            ) from None
        except RuntimeError:
            raise HTTPException(
                409, "Verified viewer operation unavailable"
            ) from None

        headers = {
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        }

        if isinstance(result, bytes):
            return Response(
                content=result,
                media_type=media_type,
                headers=headers,
            )

        from fastapi.responses import JSONResponse
        return JSONResponse(result, headers=headers)

    return api
