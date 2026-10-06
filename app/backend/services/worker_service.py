"""Launch isolated live-inference workers."""

from __future__ import annotations

import subprocess
import sys

from app.backend.core import paths
from app.backend.schemas.runs import RunError, RunRecord
from app.backend.services.run_service import (
    load_live_run,
    update_live_run,
)


class RunLaunchConflict(RuntimeError):
    pass


def launch_run_worker(
    run_id: str,
) -> RunRecord:
    run = load_live_run(
        run_id
    )

    if run.status != "queued":
        raise RunLaunchConflict(
            "Only queued runs can be started."
        )

    run_root = (
        paths.LIVE_RUNS_ROOT
        / run_id
    )

    log_path = (
        run_root
        / "worker.log"
    )

    update_live_run(
        run_id,
        status="running",
        stage="validating",
        progress=0.01,
    )

    try:
        with log_path.open(
            "ab"
        ) as log_stream:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    (
                        "app.backend.workers."
                        "run_worker"
                    ),
                    run_id,
                ],
                cwd=paths.PROJECT_ROOT,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    except Exception as exc:
        update_live_run(
            run_id,
            status="failed",
            stage="failed",
            progress=0.0,
            error=RunError(
                code="worker_launch_failed",
                message=str(exc),
            ),
        )

        raise

    # After process launch, the child worker owns all further
    # run-record mutations. This avoids parent/worker write races.
    return load_live_run(
        run_id
    )
