from hashlib import sha256

import pytest

from app.backend.services import frozen_test_service as service


def test_verified_frozen_summary():
    summary = service.get_frozen_test_summary()

    assert summary["milestone"] == "M9E6"
    assert summary["case_count"] == 129

    assert summary["action_counts"] == {
        "ABSTAIN_BASELINE": 85,
        "APPLY_LOCAL_FP": 44,
    }

    assert (
        summary["overall_summary"]["delta_dice_mean"]
        == -0.003755445490538544
    )

    assert (
        summary["overall_summary"]["delta_hd95_mm_mean"]
        == -0.6357815398580224
    )

    assert (
        summary["evaluation_sha256"]
        == service.EXPECTED_SHA256
    )


def test_missing_evidence_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        service,
        "EVALUATION_PATH",
        tmp_path / "missing.json",
    )

    with pytest.raises(service.FrozenEvidenceError):
        service.load_verified_evaluation()


def test_tampered_evidence_is_rejected(tmp_path, monkeypatch):
    original = service.EVALUATION_PATH.read_bytes()

    altered = tmp_path / "altered.json"
    altered.write_bytes(original + b"\n")

    assert sha256(altered.read_bytes()).hexdigest() != (
        service.EXPECTED_SHA256
    )

    monkeypatch.setattr(
        service,
        "EVALUATION_PATH",
        altered,
    )

    with pytest.raises(
        service.FrozenEvidenceError,
        match="checksum mismatch",
    ):
        service.load_verified_evaluation()
