import httpx

from app.research_ui.api_client import (
    ResearchApiClient,
)
from app.research_ui.main import (
    inspect_existing_run,
)


RUN_ID = (
    "run_11111111111111111111111111111111"
)


def test_inspector_loads_completed_run_and_result() -> None:
    seen: list[str] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        seen.append(
            request.url.path
        )

        if request.url.path.endswith(
            "/artifacts/result_json"
        ):
            return httpx.Response(
                200,
                json={
                    "action": "APPLY_LOCAL_FP",
                    "sam_used": True,
                },
            )

        if request.url.path.endswith(
            f"/runs/{RUN_ID}"
        ):
            return httpx.Response(
                200,
                json={
                    "run_id": RUN_ID,
                    "case_id": "case_test",
                    "status": "complete",
                    "stage": "complete",
                    "progress": 1.0,
                    "created_at": (
                        "2026-10-07T00:00:00Z"
                    ),
                    "updated_at": (
                        "2026-10-07T00:01:00Z"
                    ),
                    "frozen_variance_threshold": (
                        0.21133705228567123
                    ),
                    "action": "APPLY_LOCAL_FP",
                    "gate_state": "FP_ELIGIBLE",
                    "sam_used": True,
                    "available_artifacts": [
                        "result_json",
                    ],
                },
            )

        raise AssertionError(
            f"Unexpected request: {request.url.path}"
        )

    client = ResearchApiClient(
        transport=httpx.MockTransport(
            handler
        )
    )

    summary, run_payload, result_payload, downloads = (
        inspect_existing_run(
            client,
            RUN_ID,
        )
    )

    assert "APPLY_LOCAL_FP" in summary
    assert "FP_ELIGIBLE" in summary

    assert run_payload is not None
    assert (
        run_payload["status"]
        == "complete"
    )

    assert result_payload == {
        "action": "APPLY_LOCAL_FP",
        "sam_used": True,
    }

    assert "result_json" in downloads
    assert (
        f"/runs/{RUN_ID}/artifacts/result_json"
        in downloads
    )

    assert seen == [
        (
            f"/api/v1/runs/{RUN_ID}"
        ),
        (
            f"/api/v1/runs/{RUN_ID}/"
            "artifacts/result_json"
        ),
    ]

    client.close()


def test_inspector_does_not_request_result_for_running_run() -> None:
    seen: list[str] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        seen.append(
            request.url.path
        )

        return httpx.Response(
            200,
            json={
                "run_id": RUN_ID,
                "case_id": "case_test",
                "status": "running",
                "stage": "automatic_pipeline",
                "progress": 0.5,
                "created_at": (
                    "2026-10-07T00:00:00Z"
                ),
                "updated_at": (
                    "2026-10-07T00:00:30Z"
                ),
                "frozen_variance_threshold": (
                    0.21133705228567123
                ),
                "available_artifacts": [],
            },
        )

    client = ResearchApiClient(
        transport=httpx.MockTransport(
            handler
        )
    )

    summary, run_payload, result_payload, downloads = (
        inspect_existing_run(
            client,
            RUN_ID,
        )
    )

    assert "**running**" in summary
    assert run_payload is not None
    assert result_payload is None

    assert seen == [
        f"/api/v1/runs/{RUN_ID}"
    ]

    client.close()


def test_inspector_rejects_empty_run_id_without_http() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            f"HTTP should not be called: {request.url}"
        )

    client = ResearchApiClient(
        transport=httpx.MockTransport(
            handler
        )
    )

    summary, run_payload, result_payload, downloads = (
        inspect_existing_run(
            client,
            "   ",
        )
    )

    assert (
        "Enter a live run ID first."
        in summary
    )

    assert run_payload is None
    assert result_payload is None

    client.close()


def test_inspector_surfaces_unknown_run_safely() -> None:
    def handler(
        _request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "detail": (
                    "Live inference run not found"
                ),
            },
        )

    client = ResearchApiClient(
        transport=httpx.MockTransport(
            handler
        )
    )

    summary, run_payload, result_payload, downloads = (
        inspect_existing_run(
            client,
            RUN_ID,
        )
    )

    assert (
        "Live inference run not found"
        in summary
    )

    assert run_payload is None
    assert result_payload is None

    client.close()
