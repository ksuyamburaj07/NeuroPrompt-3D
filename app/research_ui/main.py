"""M11C NeuroPrompt-3D Gradio research console."""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from typing import Any

import gradio as gr

from app.research_ui.api_client import (
    DEFAULT_API_BASE_URL,
    ResearchApiClient,
    ResearchApiError,
)
from app.research_ui.formatters import (
    format_health,
    format_run,
    format_validation,
)


POLL_INTERVAL_SECONDS = 2.0
MAX_POLL_SECONDS = 4 * 60 * 60


def _error_markdown(
    title: str,
    exc: Exception,
) -> str:
    message = (
        str(exc)
        .replace("`", "'")
        .strip()
    )

    return (
        f"### {title}\n\n"
        f"**Error:** {message}"
    )


def build_demo(
    client: ResearchApiClient | None = None,
) -> gr.Blocks:
    api = (
        client
        if client is not None
        else ResearchApiClient(
            os.environ.get(
                "NEUROPROMPT_API_BASE_URL",
                DEFAULT_API_BASE_URL,
            )
        )
    )

    def check_backend() -> tuple[
        str,
        dict[str, Any] | None,
    ]:
        try:
            payload = api.health()

        except ResearchApiError as exc:
            return (
                _error_markdown(
                    "Backend unavailable",
                    exc,
                ),
                None,
            )

        return (
            format_health(
                payload
            ),
            payload,
        )

    def validate_case(
        t1n: object,
        t1c: object,
        t2w: object,
        t2f: object,
        segmentation: object | None,
    ) -> tuple[
        str,
        str,
        dict[str, Any] | None,
    ]:
        try:
            payload = api.validate_case(
                t1n=t1n,
                t1c=t1c,
                t2w=t2w,
                t2f=t2f,
                segmentation=segmentation,
            )

        except (
            ResearchApiError,
            ValueError,
        ) as exc:
            return (
                "",
                _error_markdown(
                    "Validation failed",
                    exc,
                ),
                None,
            )

        case_id = (
            payload.get(
                "case_id"
            )
            if (
                payload.get("valid")
                and payload.get(
                    "ready_for_inference"
                )
            )
            else ""
        )

        if not isinstance(
            case_id,
            str,
        ):
            case_id = ""

        return (
            case_id,
            format_validation(
                payload
            ),
            payload,
        )

    def run_case(
        case_id: str,
    ) -> Generator[
        tuple[
            str,
            str,
            dict[str, Any] | None,
            dict[str, Any] | None,
        ],
        None,
        None,
    ]:
        if not case_id:
            yield (
                "",
                (
                    "### Run blocked\n\n"
                    "Validate a complete MRI case first."
                ),
                None,
                None,
            )
            return

        run_id = ""

        try:
            current = api.create_run(
                case_id
            )

            value = current.get(
                "run_id"
            )

            if not isinstance(
                value,
                str,
            ) or not value:
                raise ResearchApiError(
                    "Backend did not return a valid run identifier."
                )

            run_id = value

            yield (
                run_id,
                format_run(
                    current
                ),
                current,
                None,
            )

            current = api.execute_run(
                run_id
            )

            yield (
                run_id,
                format_run(
                    current
                ),
                current,
                None,
            )

            deadline = (
                time.monotonic()
                + MAX_POLL_SECONDS
            )

            previous_signature: tuple[
                object,
                ...,
            ] | None = None

            while current.get(
                "status"
            ) not in {
                "complete",
                "failed",
            }:
                if (
                    time.monotonic()
                    >= deadline
                ):
                    raise ResearchApiError(
                        "Inference monitoring exceeded the research-console timeout."
                    )

                time.sleep(
                    POLL_INTERVAL_SECONDS
                )

                current = api.get_run(
                    run_id
                )

                signature = (
                    current.get(
                        "status"
                    ),
                    current.get(
                        "stage"
                    ),
                    current.get(
                        "progress"
                    ),
                    current.get(
                        "action"
                    ),
                    current.get(
                        "hotspot_variance"
                    ),
                )

                if (
                    signature
                    != previous_signature
                ):
                    previous_signature = (
                        signature
                    )

                    yield (
                        run_id,
                        format_run(
                            current
                        ),
                        current,
                        None,
                    )

            result_payload = None

            if (
                current.get(
                    "status"
                )
                == "complete"
                and "result_json"
                in current.get(
                    "available_artifacts",
                    [],
                )
            ):
                result_payload = (
                    api.get_result_json(
                        run_id
                    )
                )

            yield (
                run_id,
                format_run(
                    current
                ),
                current,
                result_payload,
            )

        except (
            ResearchApiError,
            ValueError,
        ) as exc:
            yield (
                run_id,
                _error_markdown(
                    "Inference failed",
                    exc,
                ),
                (
                    {
                        "run_id": run_id,
                        "error": str(exc),
                    }
                    if run_id
                    else None
                ),
                None,
            )

    def delete_case(
        case_id: str,
    ) -> tuple[
        str,
        str,
    ]:
        if not case_id:
            return (
                "",
                (
                    "No staged live case "
                    "is currently selected."
                ),
            )

        try:
            api.delete_case(
                case_id
            )

        except (
            ResearchApiError,
            ValueError,
        ) as exc:
            return (
                case_id,
                _error_markdown(
                    "Case deletion failed",
                    exc,
                ),
            )

        return (
            "",
            (
                "### Uploaded case deleted\n\n"
                f"`{case_id}` was removed from "
                "temporary live-case storage."
            ),
        )

    with gr.Blocks(
        title=(
            "NeuroPrompt-3D Research Console"
        ),
    ) as demo:
        gr.Markdown(
            """
# NeuroPrompt-3D — Research Console

*Uncertainty-Aware Automatic Prompting for 3D Brain MRI Tumor Segmentation*

**Research prototype. Not for clinical diagnosis or patient-care decisions.**

`LIVE INFERENCE | User-uploaded MRI · New Result`

This console is a technical interface for inspecting the M11 live-inference
pipeline. It does not represent the frozen M9E final-test explorer.
"""
        )

        case_state = gr.State(
            value=""
        )

        run_state = gr.State(
            value=""
        )

        with gr.Accordion(
            "Backend / provenance",
            open=True,
        ):
            backend_button = gr.Button(
                "Check Backend"
            )

            backend_status = gr.Markdown(
                "Backend has not been checked yet."
            )

            backend_json = gr.JSON(
                label="Backend metadata",
            )

        gr.Markdown(
            "## MRI inputs"
        )

        with gr.Row():
            t1n = gr.File(
                label="T1n",
                file_types=[
                    ".nii",
                    ".nii.gz",
                ],
                type="filepath",
            )

            t1c = gr.File(
                label="T1c",
                file_types=[
                    ".nii",
                    ".nii.gz",
                ],
                type="filepath",
            )

        with gr.Row():
            t2w = gr.File(
                label="T2w",
                file_types=[
                    ".nii",
                    ".nii.gz",
                ],
                type="filepath",
            )

            t2f = gr.File(
                label="T2f / FLAIR",
                file_types=[
                    ".nii",
                    ".nii.gz",
                ],
                type="filepath",
            )

        segmentation = gr.File(
            label=(
                "Optional ground-truth segmentation "
                "(research evaluation only)"
            ),
            file_types=[
                ".nii",
                ".nii.gz",
            ],
            type="filepath",
        )

        validate_button = gr.Button(
            "Validate Case",
            variant="primary",
        )

        validation_summary = gr.Markdown(
            "No case has been validated."
        )

        with gr.Accordion(
            "Technical validation payload",
            open=False,
        ):
            validation_json = gr.JSON(
                label="Validation JSON",
            )

        gr.Markdown(
            "## Live inference"
        )

        run_button = gr.Button(
            "Run NeuroPrompt-3D",
            variant="primary",
        )

        pipeline_status = gr.Markdown(
            "No inference run has been started."
        )

        with gr.Accordion(
            "Technical run payload",
            open=False,
        ):
            run_json = gr.JSON(
                label="Run JSON",
            )

        with gr.Accordion(
            "Frozen pipeline result metadata",
            open=False,
        ):
            result_json = gr.JSON(
                label="result.json",
            )

        gr.Markdown(
            "## Temporary upload lifecycle"
        )

        delete_button = gr.Button(
            "Delete Uploaded Case"
        )

        delete_status = gr.Markdown(
            (
                "Case deletion removes the staged MRI/GT upload. "
                "Completed live-run results remain separate temporary resources."
            )
        )

        backend_button.click(
            fn=check_backend,
            inputs=[],
            outputs=[
                backend_status,
                backend_json,
            ],
        )

        validate_button.click(
            fn=validate_case,
            inputs=[
                t1n,
                t1c,
                t2w,
                t2f,
                segmentation,
            ],
            outputs=[
                case_state,
                validation_summary,
                validation_json,
            ],
        )

        run_button.click(
            fn=run_case,
            inputs=[
                case_state,
            ],
            outputs=[
                run_state,
                pipeline_status,
                run_json,
                result_json,
            ],
        )

        delete_button.click(
            fn=delete_case,
            inputs=[
                case_state,
            ],
            outputs=[
                case_state,
                delete_status,
            ],
        )

    demo.queue()

    return demo


def main() -> None:
    host = os.environ.get(
        "NEUROPROMPT_GRADIO_HOST",
        "127.0.0.1",
    )

    port = int(
        os.environ.get(
            "NEUROPROMPT_GRADIO_PORT",
            "7860",
        )
    )

    demo = build_demo()

    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
    )


if __name__ == "__main__":
    main()
