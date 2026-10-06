import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import app.backend.services.execution_service as execution

from app.backend.core import paths
from app.backend.main import app
from app.backend.services.run_service import (
    create_live_run,
    load_live_run,
)
from src.uncertainty.mc_dropout import (
    derive_case_seed,
)


client = TestClient(app)


@pytest.fixture
def isolated_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    live_cases = (
        tmp_path
        / "live_cases"
    )

    live_runs = (
        tmp_path
        / "live_runs"
    )

    monkeypatch.setattr(
        paths,
        "LIVE_CASES_ROOT",
        live_cases,
    )

    monkeypatch.setattr(
        paths,
        "LIVE_RUNS_ROOT",
        live_runs,
    )

    return (
        live_cases,
        live_runs,
    )


def _nifti_bytes(
    tmp_path: Path,
    name: str,
) -> bytes:
    values = np.ones(
        (6, 5, 4),
        dtype=np.float32,
    )

    path = tmp_path / name

    nib.save(
        nib.Nifti1Image(
            values,
            np.diag(
                [
                    1.25,
                    1.5,
                    2.0,
                    1.0,
                ]
            ),
        ),
        path,
    )

    return path.read_bytes()


def _create_case(
    tmp_path: Path,
) -> str:
    files = {
        field: (
            f"{field}.nii.gz",
            _nifti_bytes(
                tmp_path,
                f"{field}.nii.gz",
            ),
            "application/gzip",
        )
        for field in (
            "t1n",
            "t1c",
            "t2w",
            "t2f",
        )
    }

    response = client.post(
        "/api/v1/cases/validate",
        files=files,
    )

    payload = response.json()

    assert payload["valid"] is True

    return payload["case_id"]


def _install_fake_models_and_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_bundle = SimpleNamespace(
        model=torch.nn.Identity(),
        config=SimpleNamespace(),
        checkpoint_sha256=(
            "b" * 64
        ),
        epoch=12,
        validation_loss=0.27398208065049634,
    )

    sam_bundle = SimpleNamespace(
        model=torch.nn.Identity(),
        source_commit=(
            "f3de1fa10da98e46f49f176773d2b1e306ba131f"
        ),
        checkpoint_sha256=(
            "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c"
        ),
        registry_key="vit_b_ori",
    )

    monkeypatch.setattr(
        execution,
        "load_baseline_checkpoint",
        lambda *_args, **_kwargs:
            baseline_bundle,
    )

    monkeypatch.setattr(
        execution,
        "load_sam_checkpoint",
        lambda *_args, **_kwargs:
            sam_bundle,
    )

    monkeypatch.setattr(
        execution,
        "resolve_execution_device",
        lambda:
            torch.device("cpu"),
    )

    def fake_pipeline(
        *,
        raw_mri_czyx,
        case_id,
        **_kwargs,
    ):
        shape = tuple(
            int(value)
            for value
            in raw_mri_czyx.shape[1:]
        )

        baseline = np.zeros(
            shape,
            dtype=bool,
        )

        baseline[
            1:3,
            1:4,
            1:5,
        ] = True

        probability = (
            baseline.astype(
                np.float32
            )
            * 0.8
        )

        final = baseline.copy()

        removal = np.zeros_like(
            baseline
        )

        support = np.zeros_like(
            baseline
        )

        return SimpleNamespace(
            action="ABSTAIN_BASELINE",
            semantic_abstention_condition=None,
            gate_state="ABSTAIN",
            case_seed=derive_case_seed(
                case_id
            ),
            mc_pass_hashes=tuple(
                f"{index:064x}"
                for index
                in range(10)
            ),
            hotspot_zyx=(1, 2, 3),
            hotspot_variance=0.1,
            fp_prompt_zyx=None,
            fp_prompt_model_xyz=None,
            sam_used=False,
            baseline_probability_zyx=probability,
            baseline_mask_zyx=baseline,
            final_mask_zyx=final,
            removal_mask_zyx=removal,
            support_mask_zyx=support,
        )

    monkeypatch.setattr(
        execution,
        "run_final_automatic_pipeline",
        fake_pipeline,
    )


def test_execute_live_run_persists_result_artifacts(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, live_runs = isolated_runtime

    _install_fake_models_and_pipeline(
        monkeypatch
    )

    case_id = _create_case(
        tmp_path
    )

    run = create_live_run(
        case_id
    )

    execution.execute_live_run(
        run.run_id
    )

    completed = load_live_run(
        run.run_id
    )

    assert completed.status == "complete"
    assert completed.stage == "complete"
    assert completed.progress == 1.0

    assert (
        completed.execution_device
        == "cpu"
    )

    assert completed.inference_case_id is not None

    assert (
        completed.inference_case_id
        .startswith(
            "live_sha256_"
        )
    )

    assert completed.mc_case_seed == (
        derive_case_seed(
            completed.inference_case_id
        )
    )

    assert completed.mc_case_seed != (
        derive_case_seed(
            case_id
        )
    )

    assert (
        completed.action
        == "ABSTAIN_BASELINE"
    )

    assert completed.sam_used is False

    assert (
        completed.sam_refinement_skipped
        is True
    )

    run_root = (
        live_runs
        / run.run_id
    )

    for relative in (
        completed.artifacts.values()
    ):
        assert (
            run_root
            / relative
        ).is_file()

    result_payload = json.loads(
        (
            run_root
            / completed.artifacts[
                "result_json"
            ]
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        result_payload[
            "inference_case_id"
        ]
        == completed.inference_case_id
    )

    assert (
        result_payload[
            "case_seed"
        ]
        == completed.mc_case_seed
    )

    final_nifti = nib.load(
        run_root
        / completed.artifacts[
            "final_mask_nifti"
        ]
    )

    assert final_nifti.shape == (
        6,
        5,
        4,
    )

    # A run ID represents one immutable execution attempt.
    with pytest.raises(
        RuntimeError,
        match="Only queued or running",
    ):
        execution.execute_live_run(
            run.run_id
        )

    unchanged = load_live_run(
        run.run_id
    )

    assert unchanged.status == "complete"
    assert unchanged.stage == "complete"


def test_execute_live_run_records_failure(
    tmp_path: Path,
    isolated_runtime: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_models_and_pipeline(
        monkeypatch
    )

    def fail_pipeline(
        **_kwargs,
    ):
        raise RuntimeError(
            "synthetic execution failure"
        )

    monkeypatch.setattr(
        execution,
        "run_final_automatic_pipeline",
        fail_pipeline,
    )

    case_id = _create_case(
        tmp_path
    )

    run = create_live_run(
        case_id
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic execution failure",
    ):
        execution.execute_live_run(
            run.run_id
        )

    failed = load_live_run(
        run.run_id
    )

    assert failed.status == "failed"
    assert failed.stage == "failed"

    assert failed.error is not None

    assert (
        failed.error.code
        == "RuntimeError"
    )


def test_live_inference_identity_is_content_deterministic() -> None:
    raw = torch.arange(
        4 * 3 * 4 * 5,
        dtype=torch.float32,
    ).reshape(
        4,
        3,
        4,
        5,
    )

    affine = np.eye(
        4,
        dtype=np.float64,
    )

    first = (
        execution._derive_inference_case_id(
            raw,
            affine,
        )
    )

    replay = (
        execution._derive_inference_case_id(
            raw.clone(),
            affine.copy(),
        )
    )

    assert first == replay
    assert first.startswith(
        "live_sha256_"
    )

    changed_voxel = raw.clone()

    changed_voxel[
        0,
        0,
        0,
        0,
    ] += 1.0

    assert (
        execution._derive_inference_case_id(
            changed_voxel,
            affine,
        )
        != first
    )

    changed_affine = affine.copy()
    changed_affine[
        0,
        3,
    ] = 1.0

    assert (
        execution._derive_inference_case_id(
            raw,
            changed_affine,
        )
        != first
    )
