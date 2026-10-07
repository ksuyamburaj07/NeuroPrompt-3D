import gradio as gr

from app.research_ui.api_client import (
    ResearchApiClient,
)
from app.research_ui.formatters import (
    format_health,
    format_run,
    format_validation,
)
from app.research_ui.main import (
    build_demo,
)


def test_format_health_exposes_provenance() -> None:
    rendered = format_health(
        {
            "status": "ok",
            "application": "NeuroPrompt-3D",
            "application_version": "0.1.0-m11",
            "frozen_research_core_commit": "66edbba",
            "m9e_read_only": True,
        }
    )

    assert "NeuroPrompt-3D" in rendered
    assert "66edbba" in rendered
    assert "M9E read-only" in rendered


def test_format_validation_exposes_geometry() -> None:
    rendered = format_validation(
        {
            "valid": True,
            "ready_for_inference": True,
            "case_id": "case_test",
            "modalities": {
                field: {
                    "valid": True,
                    "research_core_name": core,
                }
                for field, core in {
                    "t1n": "T1",
                    "t1c": "T1ce",
                    "t2w": "T2",
                    "t2f": "FLAIR",
                }.items()
            },
            "geometry": {
                "shape_xyz": [
                    182,
                    218,
                    182,
                ],
                "tensor_shape_dhw": [
                    182,
                    218,
                    182,
                ],
                "voxel_spacing_xyz": [
                    1.0,
                    1.0,
                    1.0,
                ],
                "affine_tolerance": 1e-5,
            },
            "ground_truth": {
                "provided": False,
                "valid": None,
                "labels": None,
            },
            "errors": [],
        }
    )

    assert "Validation — READY" in rendered
    assert "FLAIR" in rendered
    assert "182, 218, 182" in rendered
    assert "1e-05" in rendered


def test_format_run_exposes_frozen_decision() -> None:
    rendered = format_run(
        {
            "run_id": "run_test",
            "status": "complete",
            "stage": "complete",
            "progress": 1.0,
            "execution_device": "cpu",
            "action": "APPLY_LOCAL_FP",
            "gate_state": "FP_ELIGIBLE",
            "frozen_variance_threshold": (
                0.21133705228567123
            ),
            "hotspot_zyx": [
                71,
                98,
                50,
            ],
            "hotspot_variance": (
                0.21728354692459106
            ),
            "mc_case_seed": 1285324160,
            "fp_prompt_zyx": [
                71,
                98,
                50,
            ],
            "fp_prompt_model_xyz": [
                79,
                62,
                33,
            ],
            "sam_used": True,
            "sam_refinement_skipped": False,
            "semantic_abstention_condition": None,
            "inference_case_id": (
                "live_sha256_test"
            ),
            "available_artifacts": [
                "result_json",
                "final_mask_nifti",
            ],
            "error": None,
        }
    )

    assert "APPLY_LOCAL_FP" in rendered
    assert "FP_ELIGIBLE" in rendered
    assert "0.21133705228567123" in rendered
    assert "1285324160" in rendered
    assert "final_mask_nifti" in rendered


def test_build_demo_returns_gradio_blocks() -> None:
    client = ResearchApiClient(
        base_url=(
            "http://127.0.0.1:9999/api/v1"
        )
    )

    demo = build_demo(
        client
    )

    assert isinstance(
        demo,
        gr.Blocks,
    )

    config = demo.get_config_file()

    labels = {
        component.get(
            "props",
            {},
        ).get(
            "label"
        )
        for component
        in config.get(
            "components",
            [],
        )
    }

    assert (
        "Existing live run ID"
        in labels
    )

    assert (
        "Existing run JSON"
        in labels
    )

    assert (
        "Existing result JSON"
        in labels
    )

    client.close()


def test_format_artifact_downloads_completed_run() -> None:
    from app.research_ui.formatters import (
        format_artifact_downloads,
    )

    rendered = format_artifact_downloads(
        {
            "status": "complete",
            "available_artifacts": [
                "final_mask_nifti",
                "result_json",
            ],
        },
        {
            "final_mask_nifti": (
                "http://127.0.0.1:8000/final"
            ),
            "result_json": (
                "http://127.0.0.1:8000/result"
            ),
        },
    )

    assert "Artifact downloads" in rendered
    assert "final_mask_nifti" in rendered
    assert "http://127.0.0.1:8000/final" in rendered
    assert "result_json" in rendered


def test_format_artifact_downloads_running_run() -> None:
    from app.research_ui.formatters import (
        format_artifact_downloads,
    )

    rendered = format_artifact_downloads(
        {
            "status": "running",
            "available_artifacts": [],
        },
        {},
    )

    assert (
        "after the live inference run completes"
        in rendered
    )
