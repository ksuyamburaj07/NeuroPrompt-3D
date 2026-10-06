"""Fresh-process entry point for one live NeuroPrompt-3D run."""

import argparse
import os
import traceback

from app.backend.services.execution_service import (
    execute_live_run,
)
from app.backend.services.run_service import (
    update_live_run,
)


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run_id",
    )

    args = parser.parse_args()

    try:
        # Once launched, the worker process owns all run-record
        # mutations until execution finishes or fails.
        update_live_run(
            args.run_id,
            worker_pid=os.getpid(),
        )

        execute_live_run(
            args.run_id
        )

    except Exception:
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
