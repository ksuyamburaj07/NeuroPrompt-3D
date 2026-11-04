"""Frozen NeuroPrompt-3D live inference worker.

Uses the original M11 execution_service without changing
the scientific pipeline, model weights, or decision policy.
"""

import modal

app = modal.App("neuroprompt3d-live-execution")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "monai==1.6.0",
        "numpy==2.4.6",
        "nibabel==5.4.2",
        "scipy==1.17.1",
        "SimpleITK==2.5.6",
        "Pillow==12.3.0",
        "scikit-image==0.26.0",
        "fastapi==0.142.2",
    )
    .add_local_dir("src", "/root/src")
    .add_local_dir("app/backend/core", "/root/app/backend/core")
    .add_local_dir("app/backend/schemas", "/root/app/backend/schemas")
    .add_local_dir("app/backend/services", "/root/app/backend/services")
)

assets = modal.Volume.from_name("np3d-research-assets")
runtime = modal.Volume.from_name("np3d-live-runtime")

CASE_ID = "case_6d362b430e614eb1a483f553aa234ba0"

BASELINE = (
    "/data/Projects/NeuroPrompt3D/Models/best_baseline.pt"
)
SAM_WEIGHTS = (
    "/data/Projects/Models/SAM-Med3D/sam_med3d_turbo.pth"
)
SAM_BUNDLE = (
    "/data/Projects/External/SAM-Med3D-frozen.bundle"
)
SAM_COMMIT = "f3de1fa10da98e46f49f176773d2b1e306ba131f"

MOUNTS = {
    "/data": assets.with_mount_options(read_only=True),
    "/root/app/backend/runtime": runtime,
}


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=3600,
    max_containers=1,
    volumes=MOUNTS,
)
def execute_cloud_run(run_id: str):
    import os
    import re
    import subprocess

    from pathlib import Path

    import torch

    if not re.fullmatch(r"run_[0-9a-f]{32}", run_id):
        raise ValueError("Invalid cloud run identifier")

    torch.set_num_threads(2)
    runtime.reload()

    # Set frozen model paths before backend imports.
    os.environ["NEUROPROMPT_SAM_SOURCE_ROOT"] = (
        "/root/sam-med3d-frozen"
    )
    os.environ["NEUROPROMPT_SAM_CHECKPOINT"] = SAM_WEIGHTS

    from app.backend.services.run_service import load_live_run

    record = load_live_run(run_id)

    if record.status != "queued":
        raise RuntimeError(
            "Run already claimed; refusing duplicate execution"
        )

    from app.backend.services.run_service import update_live_run

    update_live_run(
        run_id,
        status="running",
        stage="validating",
        progress=0.01,
    )
    runtime.commit()

    # Cloud worker failure recovery.
    try:
        # Reconstruct the same frozen source used in the
        # successfully verified persistence regression.
        source_root = Path("/root/sam-med3d-frozen")

        subprocess.run(
            ["git", "clone", SAM_BUNDLE, str(source_root)],
            check=True,
            capture_output=True,
            text=True,
        )

        subprocess.run(
            [
                "git", "-C", str(source_root),
                "checkout", "--detach", SAM_COMMIT,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        actual_commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()

        if actual_commit != SAM_COMMIT:
            raise RuntimeError("Frozen SAM source mismatch")

        # Preserve the original application's checkpoint contract.
        checkpoint = Path(
            "/root/checkpoints/"
            "M7K_C_Full_Baseline_Training/best_baseline.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.symlink_to(BASELINE)

        from app.backend.services import execution_service

        original_update = execution_service.update_live_run

        def durable_update(*args, **kwargs):
            result = original_update(*args, **kwargs)
            runtime.commit()
            return result

        # Every original stage update becomes visible to the
        # separate browser polling service.
        execution_service.update_live_run = durable_update

        try:
            execution_service.execute_live_run(run_id)
        finally:
            runtime.commit()

        completed = load_live_run(run_id)

        if completed.status != "complete":
            raise RuntimeError("Inference did not complete")

        return {
            "run_id": completed.run_id,
            "case_id": completed.case_id,
            "status": completed.status,
            "action": completed.action,
            "artifact_count": len(completed.artifacts),
        }
    except Exception as exc:
        from app.backend.schemas.runs import RunError
        current = load_live_run(run_id)
        if current.status == 'running':
            update_live_run(
                run_id,
                status='failed',
                stage='failed',
                error=RunError(
                    code=type(exc).__name__,
                    message=str(exc)[:400] or 'Worker failed',
                ),
            )
            runtime.commit()
        raise


@app.function(
    image=image,
    cpu=0.25,
    memory=2048,
    timeout=180,
    volumes=MOUNTS,
)
def verify_execution_environment():
    from pathlib import Path

    from app.backend.services.case_service import load_live_case
    from app.backend.services.execution_service import execute_live_run

    runtime.reload()

    case = load_live_case(CASE_ID)

    assert case.valid is True
    assert case.ready_for_inference is True
    assert callable(execute_live_run)

    for path in (BASELINE, SAM_WEIGHTS, SAM_BUNDLE):
        assert Path(path).is_file(), path

    return {
        "case_id": CASE_ID,
        "case_ready": True,
        "frozen_assets_found": 3,
        "original_executor_imported": True,
    }


@app.local_entrypoint()
def verify():
    result = verify_execution_environment.remote()

    assert result["case_ready"]
    assert result["frozen_assets_found"] == 3
    assert result["original_executor_imported"]

    print("PASS: Browser-uploaded MRI case available")
    print("PASS: Original frozen model assets available")
    print("PASS: Original scientific executor imports")
    print(result)
