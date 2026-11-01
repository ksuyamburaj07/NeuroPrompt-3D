import pytest

from app.backend.services import frozen_test_service as service


def test_default_case_page():
    result = service.get_frozen_test_cases()

    assert result["milestone"] == "M9E6"
    assert result["total_cases"] == 129
    assert result["total_matching"] == 129
    assert len(result["cases"]) == 20

    indices = [
        row["case_index"] for row in result["cases"]
    ]

    assert indices == list(range(1, 21))


def test_last_case_page():
    result = service.get_frozen_test_cases(
        offset=120,
        limit=20,
    )

    indices = [
        row["case_index"] for row in result["cases"]
    ]

    assert indices == list(range(121, 130))
    assert len(result["cases"]) == 9


def test_action_filters():
    applied = service.get_frozen_test_cases(
        action="APPLY_LOCAL_FP",
        limit=129,
    )

    abstained = service.get_frozen_test_cases(
        action="ABSTAIN_BASELINE",
        limit=129,
    )

    assert applied["total_matching"] == 44
    assert abstained["total_matching"] == 85

    assert all(
        row["action"] == "APPLY_LOCAL_FP"
        for row in applied["cases"]
    )

    assert all(
        row["action"] == "ABSTAIN_BASELINE"
        for row in abstained["cases"]
    )


def test_public_list_excludes_raw_case_ids():
    result = service.get_frozen_test_cases(limit=129)

    assert all(
        "case_id" not in row
        for row in result["cases"]
    )

    assert len({
        row["case_index"]
        for row in result["cases"]
    }) == 129


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        service.get_frozen_test_cases(limit=0)

    with pytest.raises(ValueError):
        service.get_frozen_test_cases(offset=-1)

    with pytest.raises(ValueError):
        service.get_frozen_test_cases(action="UNKNOWN")


def test_tampered_evidence_rejected(tmp_path, monkeypatch):
    original = service.EVALUATION_PATH.read_bytes()

    altered = tmp_path / "altered.json"
    altered.write_bytes(original + b"\n")

    monkeypatch.setattr(
        service,
        "EVALUATION_PATH",
        altered,
    )

    with pytest.raises(
        service.FrozenEvidenceError,
        match="checksum mismatch",
    ):
        service.get_frozen_test_cases()
