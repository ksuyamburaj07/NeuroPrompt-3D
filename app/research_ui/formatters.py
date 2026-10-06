"""Presentation helpers for the M11 research console."""

from __future__ import annotations

from typing import Any


def _value(
    value: object,
) -> str:
    if value is None:
        return "—"

    if isinstance(
        value,
        bool,
    ):
        return (
            "Yes"
            if value
            else "No"
        )

    return str(value)


def _coordinate(
    value: object,
) -> str:
    if not isinstance(
        value,
        list,
    ):
        return "—"

    return (
        "["
        + ", ".join(
            str(item)
            for item in value
        )
        + "]"
    )


def format_health(
    payload: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "### Backend",
            "",
            f"- Status: **{_value(payload.get('status'))}**",
            (
                "- Application: "
                f"`{_value(payload.get('application'))}`"
            ),
            (
                "- Version: "
                f"`{_value(payload.get('application_version'))}`"
            ),
            (
                "- Frozen core: "
                f"`{_value(payload.get('frozen_research_core_commit'))}`"
            ),
            (
                "- M9E read-only: "
                f"**{_value(payload.get('m9e_read_only'))}**"
            ),
        ]
    )


def format_validation(
    payload: dict[str, Any],
) -> str:
    valid = bool(
        payload.get("valid")
    )

    ready = bool(
        payload.get(
            "ready_for_inference"
        )
    )

    state = (
        "READY"
        if valid and ready
        else "INVALID"
    )

    lines = [
        f"### Validation — {state}",
        "",
        (
            "- Case ID: "
            f"`{_value(payload.get('case_id'))}`"
        ),
    ]

    modalities = payload.get(
        "modalities"
    )

    if isinstance(
        modalities,
        dict,
    ):
        lines.extend(
            [
                "",
                "**Modalities**",
            ]
        )

        for field in (
            "t1n",
            "t1c",
            "t2w",
            "t2f",
        ):
            item = modalities.get(
                field,
                {},
            )

            if not isinstance(
                item,
                dict,
            ):
                item = {}

            lines.append(
                (
                    f"- {field}: "
                    f"{_value(item.get('valid'))} "
                    f"(`{_value(item.get('research_core_name'))}`)"
                )
            )

    geometry = payload.get(
        "geometry"
    )

    if isinstance(
        geometry,
        dict,
    ):
        lines.extend(
            [
                "",
                "**Geometry**",
                (
                    "- Shape XYZ: "
                    f"`{_coordinate(geometry.get('shape_xyz'))}`"
                ),
                (
                    "- Tensor DHW: "
                    f"`{_coordinate(geometry.get('tensor_shape_dhw'))}`"
                ),
                (
                    "- Voxel spacing XYZ: "
                    f"`{_coordinate(geometry.get('voxel_spacing_xyz'))}`"
                ),
                (
                    "- Affine tolerance: "
                    f"`{_value(geometry.get('affine_tolerance'))}`"
                ),
            ]
        )

    ground_truth = payload.get(
        "ground_truth"
    )

    if isinstance(
        ground_truth,
        dict,
    ):
        lines.extend(
            [
                "",
                "**Optional ground truth**",
                (
                    "- Provided: "
                    f"{_value(ground_truth.get('provided'))}"
                ),
                (
                    "- Valid: "
                    f"{_value(ground_truth.get('valid'))}"
                ),
                (
                    "- Labels: "
                    f"`{_coordinate(ground_truth.get('labels'))}`"
                ),
            ]
        )

    errors = payload.get(
        "errors"
    )

    if isinstance(
        errors,
        list,
    ) and errors:
        lines.extend(
            [
                "",
                "**Validation issues**",
            ]
        )

        for item in errors:
            if not isinstance(
                item,
                dict,
            ):
                continue

            code = _value(
                item.get("code")
            )

            message = _value(
                item.get("message")
            )

            lines.append(
                f"- `{code}` — {message}"
            )

    return "\n".join(
        lines
    )


def format_run(
    payload: dict[str, Any],
) -> str:
    progress = payload.get(
        "progress"
    )

    if isinstance(
        progress,
        (
            int,
            float,
        ),
    ):
        progress_text = (
            f"{100.0 * float(progress):.0f}%"
        )

    else:
        progress_text = "—"

    artifacts = payload.get(
        "available_artifacts"
    )

    if not isinstance(
        artifacts,
        list,
    ):
        artifacts = []

    lines = [
        "### Pipeline",
        "",
        (
            "- Run ID: "
            f"`{_value(payload.get('run_id'))}`"
        ),
        (
            "- Status: "
            f"**{_value(payload.get('status'))}**"
        ),
        (
            "- Stage: "
            f"**{_value(payload.get('stage'))}**"
        ),
        (
            "- Progress: "
            f"**{progress_text}**"
        ),
        (
            "- Device: "
            f"`{_value(payload.get('execution_device'))}`"
        ),
        "",
        "**Frozen decision state**",
        (
            "- Action: "
            f"**{_value(payload.get('action'))}**"
        ),
        (
            "- Gate: "
            f"`{_value(payload.get('gate_state'))}`"
        ),
        (
            "- Frozen variance threshold: "
            f"`{_value(payload.get('frozen_variance_threshold'))}`"
        ),
        (
            "- Hotspot ZYX: "
            f"`{_coordinate(payload.get('hotspot_zyx'))}`"
        ),
        (
            "- Hotspot variance: "
            f"`{_value(payload.get('hotspot_variance'))}`"
        ),
        (
            "- MC case seed: "
            f"`{_value(payload.get('mc_case_seed'))}`"
        ),
        (
            "- FP prompt ZYX: "
            f"`{_coordinate(payload.get('fp_prompt_zyx'))}`"
        ),
        (
            "- FP prompt model XYZ: "
            f"`{_coordinate(payload.get('fp_prompt_model_xyz'))}`"
        ),
        (
            "- SAM used: "
            f"**{_value(payload.get('sam_used'))}**"
        ),
        (
            "- SAM refinement skipped: "
            f"{_value(payload.get('sam_refinement_skipped'))}"
        ),
        (
            "- Semantic abstention: "
            f"`{_value(payload.get('semantic_abstention_condition'))}`"
        ),
        (
            "- Inference identity: "
            f"`{_value(payload.get('inference_case_id'))}`"
        ),
    ]

    if artifacts:
        lines.extend(
            [
                "",
                "**Available artifacts**",
                *[
                    f"- `{name}`"
                    for name in artifacts
                ],
            ]
        )

    error = payload.get(
        "error"
    )

    if isinstance(
        error,
        dict,
    ):
        lines.extend(
            [
                "",
                "**Run error**",
                (
                    f"- `{_value(error.get('code'))}` — "
                    f"{_value(error.get('message'))}"
                ),
            ]
        )

    return "\n".join(
        lines
    )
