"""Read-only access to the verified M9E6 final-test evaluation."""

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from app.backend.core.paths import FROZEN_M9E_ROOT


EVALUATION_PATH = (
    FROZEN_M9E_ROOT
    / "2026-09-29"
    / "M9E6_FINAL_TEST_EVALUATION"
    / "M9E6_FINAL_TEST_EVALUATION_2026_09_29.json"
)

EXPECTED_SHA256 = (
    "1771b3bf60a59938c6c63c9b07c2b26ee8cb325"
    "b3e0a9c2766be1b9890f08909"
)


class FrozenEvidenceError(RuntimeError):
    """Frozen evidence is missing, altered, or invalid."""


def load_verified_evaluation() -> dict:
    """Read and verify the immutable M9E6 JSON."""
    try:
        raw = EVALUATION_PATH.read_bytes()
    except OSError as exc:
        raise FrozenEvidenceError(
            "Frozen M9E6 evaluation is unavailable."
        ) from exc

    digest = sha256(raw).hexdigest()

    if digest != EXPECTED_SHA256:
        raise FrozenEvidenceError(
            "Frozen M9E6 evaluation checksum mismatch."
        )

    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise FrozenEvidenceError(
            "Frozen M9E6 evaluation cannot be decoded."
        ) from exc

    if not isinstance(document, dict):
        raise FrozenEvidenceError("Invalid evaluation structure.")

    cases = document.get("case_results")

    if (
        document.get("milestone") != "M9E6"
        or document.get("status") != "final_test_evaluation"
        or document.get("case_count") != 129
        or not isinstance(cases, list)
        or len(cases) != 129
    ):
        raise FrozenEvidenceError(
            "Frozen evaluation identity is inconsistent."
        )

    return document


def get_frozen_test_summary() -> dict:
    """Expose recorded metrics without changing frozen results."""
    document = load_verified_evaluation()
    cases = document["case_results"]

    actions = Counter(row["action"] for row in cases)

    if actions != {
        "APPLY_LOCAL_FP": 44,
        "ABSTAIN_BASELINE": 85,
    }:
        raise FrozenEvidenceError(
            "Frozen policy distribution is inconsistent."
        )

    if len({row["case_id"] for row in cases}) != 129:
        raise FrozenEvidenceError(
            "Frozen case identities are inconsistent."
        )

    return {
        "milestone": document["milestone"],
        "status": document["status"],
        "case_count": document["case_count"],
        "evaluation_timestamp_utc": (
            document["evaluation_timestamp_utc"]
        ),
        "evaluation_sha256": EXPECTED_SHA256,
        "overall_summary": document["overall_summary"],
        "improvement_counts": document["improvement_counts"],
        "metric_contract": document["metric_contract"],
        "scientific_boundary": document["scientific_boundary"],
        "action_counts": dict(actions),
    }


PUBLIC_CASE_FIELDS = (
    "case_index",
    "action",
    "gate_state",
    "sam_used",
    "hotspot_variance",
    "baseline_dice",
    "final_dice",
    "delta_dice",
    "baseline_iou",
    "final_iou",
    "delta_iou",
    "baseline_hd95_mm",
    "final_hd95_mm",
    "delta_hd95_mm",
)


def get_frozen_test_cases(
    *,
    offset: int = 0,
    limit: int = 20,
    action: str | None = None,
) -> dict:
    """Return a filtered page of verified frozen case measurements."""

    if offset < 0 or not 1 <= limit <= 129:
        raise ValueError("Invalid pagination parameters.")

    if action not in (
        None,
        "APPLY_LOCAL_FP",
        "ABSTAIN_BASELINE",
    ):
        raise ValueError("Invalid frozen policy action.")

    document = load_verified_evaluation()
    cases = sorted(
        document["case_results"],
        key=lambda row: row["case_index"],
    )

    indices = [row["case_index"] for row in cases]

    if indices != list(range(1, 130)):
        raise FrozenEvidenceError(
            "Frozen case-index sequence is inconsistent."
        )

    if action is not None:
        cases = [
            row for row in cases
            if row["action"] == action
        ]

    total_matching = len(cases)
    selected = cases[offset:offset + limit]

    public_cases = [
        {key: row[key] for key in PUBLIC_CASE_FIELDS}
        for row in selected
    ]

    return {
        "milestone": document["milestone"],
        "total_cases": document["case_count"],
        "total_matching": total_matching,
        "offset": offset,
        "limit": limit,
        "action_filter": action,
        "cases": public_cases,
    }


def get_frozen_test_case(*, case_index: int) -> dict:
    """Return one verified, public-safe frozen case."""

    if (
        not isinstance(case_index, int)
        or isinstance(case_index, bool)
        or not 1 <= case_index <= 129
    ):
        raise ValueError("Invalid frozen case index.")

    page = get_frozen_test_cases(
        offset=case_index - 1,
        limit=1,
    )

    cases = page["cases"]

    if len(cases) != 1:
        raise FrozenEvidenceError(
            "Frozen case record is unavailable."
        )

    record = cases[0]

    if record["case_index"] != case_index:
        raise FrozenEvidenceError(
            "Frozen case index is inconsistent."
        )

    return {
        "milestone": page["milestone"],
        "evaluation_sha256": EXPECTED_SHA256,
        "case": record,
    }
