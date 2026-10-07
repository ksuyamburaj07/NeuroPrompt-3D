"""HTTP client used by the M11 Gradio research console."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


DEFAULT_API_BASE_URL = (
    "http://127.0.0.1:8000/api/v1"
)


class ResearchApiError(RuntimeError):
    """User-safe error raised for backend/API failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


def _upload_path(
    value: object,
) -> Path:
    if isinstance(
        value,
        (
            str,
            Path,
        ),
    ):
        return Path(value)

    for attribute in (
        "path",
        "name",
    ):
        candidate = getattr(
            value,
            attribute,
            None,
        )

        if isinstance(
            candidate,
            (
                str,
                Path,
            ),
        ):
            return Path(candidate)

    raise ValueError(
        "Uploaded file does not expose a usable filesystem path."
    )


def _media_type(
    path: Path,
) -> str:
    if path.name.lower().endswith(
        ".nii.gz"
    ):
        return "application/gzip"

    return "application/octet-stream"


class ResearchApiClient:
    """Thin client for the public NeuroPrompt-3D M11 API."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_BASE_URL,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (
            base_url.rstrip("/")
        )

        self._client = httpx.Client(
            base_url=(
                self.base_url
                + "/"
            ),
            transport=transport,
            timeout=httpx.Timeout(
                connect=5.0,
                read=300.0,
                write=300.0,
                pool=5.0,
            ),
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = (
                self._client.request(
                    method,
                    path,
                    **kwargs,
                )
            )

        except httpx.RequestError as exc:
            raise ResearchApiError(
                "Unable to reach the NeuroPrompt-3D backend."
            ) from exc

        if response.is_error:
            detail: str | None = None

            try:
                payload = response.json()

                if isinstance(
                    payload,
                    dict,
                ):
                    value = payload.get(
                        "detail"
                    )

                    if isinstance(
                        value,
                        str,
                    ):
                        detail = value

            except ValueError:
                pass

            raise ResearchApiError(
                detail
                or (
                    "NeuroPrompt-3D backend request "
                    f"failed with HTTP {response.status_code}."
                ),
                status_code=response.status_code,
            )

        try:
            payload = response.json()

        except ValueError as exc:
            raise ResearchApiError(
                "NeuroPrompt-3D backend returned invalid JSON.",
                status_code=response.status_code,
            ) from exc

        if not isinstance(
            payload,
            dict,
        ):
            raise ResearchApiError(
                "NeuroPrompt-3D backend returned an unexpected response.",
                status_code=response.status_code,
            )

        return payload

    def health(
        self,
    ) -> dict[str, Any]:
        return self._request_json(
            "GET",
            "health",
        )

    def validate_case(
        self,
        *,
        t1n: object,
        t1c: object,
        t2w: object,
        t2f: object,
        segmentation: object | None = None,
    ) -> dict[str, Any]:
        supplied = {
            "t1n": t1n,
            "t1c": t1c,
            "t2w": t2w,
            "t2f": t2f,
        }

        missing = [
            field
            for field, value
            in supplied.items()
            if value is None
        ]

        if missing:
            raise ValueError(
                "Missing required MRI modalities: "
                + ", ".join(missing)
            )

        with ExitStack() as stack:
            files: dict[
                str,
                tuple[
                    str,
                    Any,
                    str,
                ],
            ] = {}

            for field, value in supplied.items():
                path = _upload_path(
                    value
                )

                if not path.is_file():
                    raise ValueError(
                        f"{field} upload is not available."
                    )

                stream = (
                    stack.enter_context(
                        path.open("rb")
                    )
                )

                files[field] = (
                    path.name,
                    stream,
                    _media_type(
                        path
                    ),
                )

            if segmentation is not None:
                path = _upload_path(
                    segmentation
                )

                if not path.is_file():
                    raise ValueError(
                        "Segmentation upload is not available."
                    )

                stream = (
                    stack.enter_context(
                        path.open("rb")
                    )
                )

                files[
                    "segmentation"
                ] = (
                    path.name,
                    stream,
                    _media_type(
                        path
                    ),
                )

            return self._request_json(
                "POST",
                "cases/validate",
                files=files,
            )

    def get_case(
        self,
        case_id: str,
    ) -> dict[str, Any]:
        return self._request_json(
            "GET",
            f"cases/{case_id}",
        )

    def delete_case(
        self,
        case_id: str,
    ) -> dict[str, Any]:
        return self._request_json(
            "DELETE",
            f"cases/{case_id}",
        )

    def create_run(
        self,
        case_id: str,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"cases/{case_id}/runs",
        )

    def execute_run(
        self,
        run_id: str,
    ) -> dict[str, Any]:
        return self._request_json(
            "POST",
            f"runs/{run_id}/execute",
        )

    def get_run(
        self,
        run_id: str,
    ) -> dict[str, Any]:
        return self._request_json(
            "GET",
            f"runs/{run_id}",
        )

    def get_result_json(
        self,
        run_id: str,
    ) -> dict[str, Any]:
        return self._request_json(
            "GET",
            (
                f"runs/{run_id}/"
                "artifacts/result_json"
            ),
        )

    def artifact_url(
        self,
        run_id: str,
        artifact_name: str,
    ) -> str:
        """Return the public API URL for one logical live artifact."""

        run_id = run_id.strip()
        artifact_name = artifact_name.strip()

        if not run_id:
            raise ValueError(
                "Run identifier is required."
            )

        if not artifact_name:
            raise ValueError(
                "Artifact name is required."
            )

        return (
            f"{self.base_url}/runs/"
            f"{quote(run_id, safe='')}/artifacts/"
            f"{quote(artifact_name, safe='')}"
        )
