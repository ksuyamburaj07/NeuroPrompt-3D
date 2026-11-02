"""NeuroPrompt-3D protected, provisional cortical research QC.

Original pinned renderer, original T1N and verified pial meshes.
Read-only, research-only. No clinical approval or inference.
"""

import modal

app = modal.App("neuroprompt3d-cortical-qc")

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
        "pillow==12.3.0",
        "scipy==1.17.1",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir(
        "app/backend/core",
        "/root/app/backend/core",
    )
    .add_local_dir(
        "app/backend/schemas",
        "/root/app/backend/schemas",
    )
    .add_local_dir(
        "app/backend/services",
        "/root/app/backend/services",
    )
)

assets = modal.Volume.from_name("np3d-research-assets")

SULCAL_SHA256 = {
    "lh.pial.ras.sulc-qc.ply":
        "578b352333ccb34e31b053eef79933397c39adca0259f4900d4b926c61651df7",
    "rh.pial.ras.sulc-qc.ply":
        "7f40552bbc87d18fad8155304cb96927e4aefa8a9485ce0e8e33a3c836fb6fde",
}

MESH_NAMES = frozenset({
    "lh.pial.ras.ply",
    "rh.pial.ras.ply",
    *SULCAL_SHA256,
})


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=180,
    max_containers=1,
    volumes={
        "/data": assets.with_mount_options(read_only=True)
    },
)
def verify_cortical():
    import hashlib

    from app.backend.services import pinned_cortical_qc as qc

    meta = qc.metadata()

    assert meta["source_sha256"] == qc.EXPECTED_MRI_SHA256
    assert meta["reconstruction_id"] == "m11e6b5_trial01"
    assert meta["qc_status"] == "provisional"
    assert meta["clinically_validated"] is False
    assert meta["shape_ras_xyz"] == [182, 218, 182]

    for filename in MESH_NAMES:
        path = qc.ASSETS / filename
        assert path.is_file() and not path.is_symlink()

        digest = qc.sha256(path)

        if filename in SULCAL_SHA256:
            assert digest == SULCAL_SHA256[filename]
        else:
            export = qc.EXPORTS / filename
            assert export.is_file()
            assert not export.is_symlink()
            assert digest == qc.sha256(export)

    results = {}

    for kind, plane, index in (
        ("slices", "axial", 91),
        ("slices3d", "coronal", 109),
        ("contours", "sagittal", 91),
    ):
        png = qc.render_png(kind, plane, index)

        assert png.startswith(b"\x89PNG\r\n\x1a\n")

        results[f"{kind}/{plane}/{index}"] = {
            "bytes": len(png),
            "sha256": hashlib.sha256(png).hexdigest(),
        }

    return {
        "source_sha256": meta["source_sha256"],
        "reconstruction_id": meta["reconstruction_id"],
        "qc_status": meta["qc_status"],
        "verified_mesh_count": len(MESH_NAMES),
        "renders": results,
    }


@app.local_entrypoint()
def verify():
    result = verify_cortical.remote()

    assert result["verified_mesh_count"] == 4
    assert result["qc_status"] == "provisional"
    assert len(result["renders"]) == 3

    print("PASS: Original cortical MRI SHA256")
    print("PASS: Four original cortical meshes verified")
    print("PASS: Three genuine QC image types rendered")
    print(result)


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=180,
    max_containers=1,
    volumes={
        "/data": assets.with_mount_options(read_only=True)
    },
)
@modal.asgi_app(requires_proxy_auth=True)
def cortical_api():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response

    from app.backend.services import pinned_cortical_qc as qc

    api = FastAPI(
        title="NeuroPrompt-3D Protected Cortical QC",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def no_query(request: Request):
        if request.url.query:
            raise HTTPException(
                status_code=400,
                detail="Unexpected query parameters",
            )

    @api.get("/qc/mri/metadata")
    def metadata(request: Request):
        no_query(request)
        return qc.metadata()

    @api.get("/qc/mri/{kind}/{plane}/{index}")
    def image_response(
        kind: str,
        plane: str,
        index: str,
        request: Request,
    ):
        no_query(request)

        if kind not in {"slices", "slices3d", "contours"}:
            raise HTTPException(status_code=404)

        if plane not in qc.PLANES:
            raise HTTPException(status_code=404)

        if (
            not index.isascii()
            or not index.isdecimal()
            or len(index) > 3
        ):
            raise HTTPException(status_code=400)

        position = int(index)
        axis = qc.PLANES[plane]

        if position >= qc.SHAPE[axis]:
            raise HTTPException(status_code=400)

        try:
            png = qc.render_png(kind, plane, position)
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Original cortical image unavailable",
            ) from None

        return Response(
            content=png,
            media_type="image/png",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @api.get("/qc/meshes/{filename}")
    def mesh_response(filename: str, request: Request):
        no_query(request)

        if filename not in MESH_NAMES:
            raise HTTPException(status_code=404)

        mesh = qc.ASSETS / filename

        if not mesh.is_file() or mesh.is_symlink():
            raise HTTPException(status_code=503)

        payload = mesh.read_bytes()
        digest = qc.sha256(mesh)

        if filename in SULCAL_SHA256:
            if digest != SULCAL_SHA256[filename]:
                raise HTTPException(status_code=503)
        else:
            export = qc.EXPORTS / filename

            if (
                not export.is_file()
                or export.is_symlink()
                or qc.sha256(export) != digest
            ):
                raise HTTPException(status_code=503)

        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return api
