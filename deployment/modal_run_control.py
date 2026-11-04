"""Protected owner-only live run control.

One allowlisted case; one scientific run.
No model, policy, or research artifact changes.
"""

import modal

app = modal.App("neuroprompt3d-run-control")

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
    .add_local_dir("app/backend/core", "/root/app/backend/core")
    .add_local_dir("app/backend/schemas", "/root/app/backend/schemas")
    .add_local_dir("app/backend/services", "/root/app/backend/services")
)

runtime = modal.Volume.from_name("np3d-live-runtime")

CASE_ID = "case_6d362b430e614eb1a483f553aa234ba0"

MOUNTS = {
    "/root/app/backend/runtime": runtime,
}


@app.function(
    image=image,
    cpu=0.25,
    memory=2048,
    timeout=180,
    max_containers=1,
    volumes=MOUNTS,
)
@modal.asgi_app(requires_proxy_auth=True)
def run_control_api():
    import json
    import re
    import threading
    from datetime import datetime, timezone
    from uuid import UUID, NAMESPACE_URL, uuid5

    from fastapi import FastAPI, HTTPException, Request

    from app.backend.core import paths
    from app.backend.schemas.runs import RunView
    from app.backend.services.run_service import (
        create_live_run,
        load_live_run,
    )

    api = FastAPI(
        title="Protected NeuroPrompt-3D Run Control",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    lock = threading.Lock()
    run_pattern = re.compile(r"run_[0-9a-f]{32}")

    def case_runs():
        root = paths.LIVE_RUNS_ROOT
        if not root.is_dir():
            return []

        records = []
        for item in root.iterdir():
            if (
                item.is_dir()
                and run_pattern.fullmatch(item.name)
                and (item / "run.json").is_file()
            ):
                record = load_live_run(item.name)
                if record.case_id == CASE_ID:
                    records.append(record)
        return records

    @api.post(
        "/api/v1/cases/{case_id}/runs",
        response_model=RunView,
    )
    def create_run(case_id: str, request: Request):
        if case_id != CASE_ID:
            raise HTTPException(403, "Case not authorized")

        key = request.headers.get("idempotency-key", "")

        try:
            parsed = UUID(key)
            if parsed.version != 4 or str(parsed) != key.lower():
                raise ValueError()
        except (ValueError, AttributeError):
            raise HTTPException(
                422, "Canonical UUID v4 request key required"
            ) from None

        expected_id = "run_" + uuid5(
            NAMESPACE_URL,
            f"neuroprompt3d/live-run/v1/{case_id}/{parsed}",
        ).hex

        with lock:
            runtime.reload()
            existing = case_runs()

            if existing:
                if (
                    len(existing) != 1
                    or existing[0].run_id != expected_id
                ):
                    raise HTTPException(
                        409, "This case already has a different run"
                    )
                return RunView.from_record(existing[0])

            record = create_live_run(
                case_id, idempotency_key=key
            )
            runtime.commit()
            return RunView.from_record(record)

    @api.post(
        "/api/v1/runs/{run_id}/execute",
        response_model=RunView,
    )
    def dispatch_run(run_id: str):
        if not run_pattern.fullmatch(run_id):
            raise HTTPException(422, "Invalid run identifier")

        with lock:
            runtime.reload()

            try:
                record = load_live_run(run_id)
            except FileNotFoundError:
                raise HTTPException(404, "Run not found") from None

            if record.case_id != CASE_ID:
                raise HTTPException(403, "Run not authorized")

            if record.status != "queued":
                return RunView.from_record(record)

            marker = (
                paths.LIVE_RUNS_ROOT
                / run_id
                / "dispatch.json"
            )

            if marker.exists():
                # Never dispatch again after an uncertain response.
                return RunView.from_record(record)

            marker.write_text(
                json.dumps({
                    "run_id": run_id,
                    "submitted_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }),
                encoding="utf-8",
            )
            runtime.commit()

            try:
                worker = modal.Function.from_name(
                    "neuroprompt3d-live-execution",
                    "execute_cloud_run",
                )
                worker.spawn(run_id)
            except Exception:
                # Retain marker: submission may have succeeded.
                raise HTTPException(
                    503,
                    "Dispatch outcome uncertain; inspect existing run",
                ) from None

            return RunView.from_record(record)

    return api


@app.function(
    image=image,
    cpu=0.25,
    memory=2048,
    timeout=120,
    max_containers=1,
    volumes={
        "/root/app/backend/runtime":
            runtime.with_mount_options(read_only=True),
    },
)
def verify_run_control():
    import re

    from app.backend.core import paths
    from app.backend.services.case_service import load_live_case
    from app.backend.services.run_service import load_live_run

    runtime.reload()

    case = load_live_case(CASE_ID)
    assert case.valid and case.ready_for_inference

    existing = []

    if paths.LIVE_RUNS_ROOT.exists():
        for item in paths.LIVE_RUNS_ROOT.iterdir():
            if (
                item.is_dir()
                and re.fullmatch(r"run_[0-9a-f]{32}", item.name)
                and (item / "run.json").is_file()
            ):
                record = load_live_run(item.name)
                if record.case_id == CASE_ID:
                    existing.append({
                        "run_id": record.run_id,
                        "status": record.status,
                    })

    return {
        "case_ready": True,
        "existing_runs_for_case": existing,
    }


@app.local_entrypoint()
def verify():
    result = verify_run_control.remote()
    assert result["case_ready"]
    print("PASS: Run controller can access validated case")
    print("Existing runs for this case:",
          result["existing_runs_for_case"])
