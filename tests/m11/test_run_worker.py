import sys

import app.backend.workers.run_worker as worker


def test_worker_records_own_pid_before_execution(
    monkeypatch,
) -> None:
    run_id = (
        "run_"
        + "0" * 32
    )

    events = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_worker",
            run_id,
        ],
    )

    monkeypatch.setattr(
        worker.os,
        "getpid",
        lambda: 4242,
    )

    def fake_update(
        received_run_id,
        **changes,
    ):
        events.append(
            (
                "update",
                received_run_id,
                changes,
            )
        )

    def fake_execute(
        received_run_id,
    ):
        events.append(
            (
                "execute",
                received_run_id,
            )
        )

    monkeypatch.setattr(
        worker,
        "update_live_run",
        fake_update,
    )

    monkeypatch.setattr(
        worker,
        "execute_live_run",
        fake_execute,
    )

    assert worker.main() == 0

    assert events == [
        (
            "update",
            run_id,
            {
                "worker_pid": 4242,
            },
        ),
        (
            "execute",
            run_id,
        ),
    ]
