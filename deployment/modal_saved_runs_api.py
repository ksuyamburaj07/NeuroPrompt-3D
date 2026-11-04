"""Protected, read-only NeuroPrompt-3D saved-run API.

Uses the original M11 run services and a persistent Volume.
Does not create, execute or modify research runs.
"""

import modal

app = modal.App("neuroprompt3d-saved-runs")

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

runtime = modal.Volume.from_name("np3d-live-runtime")

volume_mount = {
    "/root/app/backend/runtime":
        runtime.with_mount_options(read_only=True),
}

TEST_RUN = "run_aae795bb5d754c70ad20789724c71cc7"

EXPECTED_MASK = (
    "28343c487dc77621a21388522584d3371bf87f44aa1d730a5f72d598389e051f"
)


@app.function(
    image=image,
    cpu=0.25,
    memory=2048,
    max_containers=1,
    volumes=volume_mount,
)
def verify_saved_runs():
    import hashlib
    import numpy as np

    from app.backend.services.run_service import (
        load_live_run,
    )
    from app.backend.services.research_run_listing_service import (
        list_recent_live_runs,
    )
    from app.backend.services.artifact_service import (
        resolve_live_artifact,
    )

    record = load_live_run(TEST_RUN)
    assert record.status == "complete"
    assert record.action == "ABSTAIN_BASELINE"
    assert len(record.artifacts) == 11

    artifact = resolve_live_artifact(
        TEST_RUN, "final_mask_npy"
    )

    mask = np.load(artifact.path, allow_pickle=False)

    sha = hashlib.sha256(
        np.ascontiguousarray(
            mask, dtype=np.uint8
        ).tobytes()
    ).hexdigest()

    assert sha == EXPECTED_MASK

    records = list_recent_live_runs(limit=20)
    assert any(r.run_id == TEST_RUN for r in records)

    return {
        "verified": True,
        "run_id": record.run_id,
        "status": record.status,
        "artifact_count": len(record.artifacts),
        "mask_sha256": sha,
        "recent_run_count": len(records),
    }


@app.local_entrypoint()
def verify():
    result = verify_saved_runs.remote()
    assert result["verified"] is True
    print("PASS: Real persisted run recovered")
    print("PASS: Original research-run discovery")
    print("PASS: Original artifact checksum")
    print(result)


@app.function(
    image=image,
    cpu=0.25,
    memory=2048,
    max_containers=1,
    volumes=volume_mount,
)
@modal.asgi_app(requires_proxy_auth=True)
def saved_runs_api():
    import re

    from fastapi import (
        FastAPI,
        HTTPException,
        Query,
    )

    from app.backend.schemas.runs import RunView
    from app.backend.services.run_service import (
        load_live_run,
    )
    from app.backend.services.research_run_listing_service import (
        list_recent_live_runs,
    )

    api = FastAPI(
        title="NeuroPrompt-3D Protected Saved Runs",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    import asyncio

    read_lock = asyncio.Lock()

    @api.middleware("http")
    async def refresh_saved_runs(request, call_next):
        async with read_lock:
            runtime.reload()
            return await call_next(request)

    @api.get("/api/v1/research/runs")
    def recent_runs(
        limit: int = Query(default=20, ge=1, le=50),
    ):
        try:
            records = list_recent_live_runs(limit=limit)
        except RuntimeError:
            raise HTTPException(
                status_code=503,
                detail="Run inventory unavailable",
            ) from None

        return {
            "runs": [
                {
                    "run_id": r.run_id,
                    "status": r.status,
                    "stage": r.stage,
                    "progress": r.progress,
                    "updated_at": r.updated_at.isoformat(),
                }
                for r in records
            ],
            "count": len(records),
            "limit": limit,
        }

    @api.get(
        "/api/v1/runs/{run_id}",
        response_model=RunView,
    )
    def inspect_run(run_id: str):
        if not re.fullmatch(r"run_[0-9a-f]{32}", run_id):
            raise HTTPException(400, "Invalid run identifier")

        try:
            record = load_live_run(run_id)
        except FileNotFoundError:
            raise HTTPException(
                404, "Saved run not found"
            ) from None

        return RunView.from_record(record)

    @api.get(
        "/api/v1/runs/{run_id}/artifacts/{artifact_name}"
    )
    def download_verified_artifact(
        run_id: str,
        artifact_name: str,
    ):
        from fastapi.responses import FileResponse
        from app.backend.services.artifact_service import (
            ARTIFACT_SPECS,
            ArtifactRunNotComplete,
            ArtifactNotFound,
            ArtifactIntegrityError,
            resolve_live_artifact,
        )

        # Owner-only release checkpoint: one verified cloud run.
        allowed_run = (
            "run_ee71bd9f95e755f7909f6011d0ef7410"
        )
        allowed_case = (
            "case_6d362b430e614eb1a483f553aa234ba0"
        )

        if (
            run_id != allowed_run
            or artifact_name not in ARTIFACT_SPECS
        ):
            raise HTTPException(404, "Artifact unavailable")

        try:
            record = load_live_run(run_id)
            if record.case_id != allowed_case:
                raise HTTPException(404, "Artifact unavailable")

            artifact = resolve_live_artifact(
                run_id, artifact_name
            )
        except (FileNotFoundError, ArtifactNotFound):
            raise HTTPException(
                404, "Artifact unavailable"
            ) from None
        except ArtifactRunNotComplete:
            raise HTTPException(
                409, "Run not complete"
            ) from None
        except ArtifactIntegrityError:
            raise HTTPException(
                503, "Artifact integrity check failed"
            ) from None

        return FileResponse(
            path=artifact.path,
            filename=artifact.filename,
            media_type=artifact.media_type,
            content_disposition_type="attachment",
            headers={
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return api


@app.function(
    image=image,
    cpu=1,
    memory=4096,
    timeout=300,
    volumes=volume_mount,
)
def verify_new_cloud_output():
    import hashlib
    import json
    import nibabel as nib
    import numpy as np

    from app.backend.services.run_service import load_live_run
    from app.backend.services.artifact_service import (
        ARTIFACT_SPECS,
        resolve_live_artifact,
    )

    run_id = "run_ee71bd9f95e755f7909f6011d0ef7410"
    case_id = "case_6d362b430e614eb1a483f553aa234ba0"

    runtime.reload()
    record = load_live_run(run_id)

    assert record.case_id == case_id
    assert record.status == "complete"
    assert set(record.artifacts) == set(ARTIFACT_SPECS)
    print("PASS: All 11 approved artifacts declared")

    for name in (
        "baseline_probability",
        "baseline_mask",
        "final_mask",
        "removal_mask",
        "support_mask",
    ):
        npy = resolve_live_artifact(run_id, name + "_npy")
        nii = resolve_live_artifact(run_id, name + "_nifti")

        array_zyx = np.load(
            npy.path, mmap_mode="r", allow_pickle=False
        )
        image_xyz = np.asarray(nib.load(nii.path).dataobj)

        assert np.array_equal(
            array_zyx, image_xyz.transpose(2, 1, 0)
        ), f"{name}: NumPy/NIfTI mismatch"

        print("PASS:", name, "NumPy ↔ NIfTI")

    result_path = resolve_live_artifact(
        run_id, "result_json"
    ).path

    result = json.loads(result_path.read_text())

    assert result["run_id"] == record.run_id
    assert result["case_id"] == record.case_id
    assert result["action"] == record.action
    assert result["gate_state"] == record.gate_state
    assert result["sam_used"] == record.sam_used
    print("PASS: Result JSON matches run record")

    final_path = resolve_live_artifact(
        run_id, "final_mask_npy"
    ).path

    final = np.load(final_path, allow_pickle=False)
    digest = hashlib.sha256(
        np.ascontiguousarray(
            final, dtype=np.uint8
        ).tobytes()
    ).hexdigest()

    print("FINAL MASK SHA256:", digest)
    print("ACTION:", record.action)
    print("GATE:", record.gate_state)
    print("SAM USED:", record.sam_used)
    print("BASELINE CHECKPOINT SHA256:",
          result["baseline"]["checkpoint_sha256"])
    print("SAM CHECKPOINT SHA256:",
          result["sam"]["checkpoint_sha256"])
    print("SAM SOURCE COMMIT:",
          result["sam"]["source_commit"])

    return {"verified": True, "mask_sha256": digest}


@app.local_entrypoint()
def verify_recent():
    result = verify_new_cloud_output.remote()
    assert result["verified"]
    print("PASS: New cloud scientific output verified")
