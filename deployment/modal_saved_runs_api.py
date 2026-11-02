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

    return api
