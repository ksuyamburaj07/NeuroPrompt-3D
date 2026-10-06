from pathlib import Path

import httpx
import pytest

from app.research_ui.api_client import (
    ResearchApiClient,
    ResearchApiError,
)


def test_research_api_client_uses_public_routes() -> None:
    seen: list[
        tuple[
            str,
            str,
        ]
    ] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        seen.append(
            (
                request.method,
                request.url.path,
            )
        )

        path = request.url.path

        if path.endswith(
            "/health"
        ):
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                },
            )

        if path.endswith(
            "/runs"
        ):
            return httpx.Response(
                201,
                json={
                    "run_id": "run_test",
                },
            )

        if path.endswith(
            "/execute"
        ):
            return httpx.Response(
                202,
                json={
                    "run_id": "run_test",
                    "status": "running",
                },
            )

        if path.endswith(
            "/artifacts/result_json"
        ):
            return httpx.Response(
                200,
                json={
                    "action": "ABSTAIN_BASELINE",
                },
            )

        if (
            "/runs/"
            in path
        ):
            return httpx.Response(
                200,
                json={
                    "run_id": "run_test",
                    "status": "complete",
                },
            )

        if (
            request.method
            == "DELETE"
        ):
            return httpx.Response(
                200,
                json={
                    "deleted": True,
                },
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {path}"
        )

    client = ResearchApiClient(
        transport=httpx.MockTransport(
            handler
        )
    )

    assert client.health()[
        "status"
    ] == "ok"

    assert client.create_run(
        "case_test"
    )["run_id"] == "run_test"

    assert client.execute_run(
        "run_test"
    )["status"] == "running"

    assert client.get_run(
        "run_test"
    )["status"] == "complete"

    assert client.get_result_json(
        "run_test"
    )["action"] == "ABSTAIN_BASELINE"

    assert client.delete_case(
        "case_test"
    )["deleted"] is True

    assert seen == [
        (
            "GET",
            "/api/v1/health",
        ),
        (
            "POST",
            "/api/v1/cases/case_test/runs",
        ),
        (
            "POST",
            "/api/v1/runs/run_test/execute",
        ),
        (
            "GET",
            "/api/v1/runs/run_test",
        ),
        (
            "GET",
            (
                "/api/v1/runs/run_test/"
                "artifacts/result_json"
            ),
        ),
        (
            "DELETE",
            "/api/v1/cases/case_test",
        ),
    ]


def test_research_api_client_sends_multipart_uploads(
    tmp_path: Path,
) -> None:
    files = {}

    for field in (
        "t1n",
        "t1c",
        "t2w",
        "t2f",
    ):
        path = (
            tmp_path
            / f"{field}.nii.gz"
        )

        path.write_bytes(
            field.encode(
                "utf-8"
            )
        )

        files[field] = path

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert (
            request.method
            == "POST"
        )

        assert (
            request.url.path
            == "/api/v1/cases/validate"
        )

        content_type = request.headers[
            "content-type"
        ]

        assert content_type.startswith(
            "multipart/form-data;"
        )

        body = request.read()

        for field in files:
            assert (
                f'name="{field}"'
                .encode("utf-8")
                in body
            )

        return httpx.Response(
            200,
            json={
                "valid": True,
                "ready_for_inference": True,
                "case_id": "case_test",
            },
        )

    client = ResearchApiClient(
        transport=httpx.MockTransport(
            handler
        )
    )

    payload = client.validate_case(
        **files
    )

    assert (
        payload["case_id"]
        == "case_test"
    )


def test_research_api_client_surfaces_backend_detail() -> None:
    def handler(
        _request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "detail": (
                    "Live case cannot be deleted "
                    "while an inference run is queued or running."
                ),
            },
        )

    client = ResearchApiClient(
        transport=httpx.MockTransport(
            handler
        )
    )

    with pytest.raises(
        ResearchApiError,
        match="cannot be deleted",
    ) as exc:
        client.delete_case(
            "case_test"
        )

    assert (
        exc.value.status_code
        == 409
    )
