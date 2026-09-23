"""Frozen M9C3 automatic action policy.

This module contains only the pure decision semantics frozen before
fresh-holdout access. It performs no model inference and opens no data.
"""

from __future__ import annotations

import math


FROZEN_VARIANCE_THRESHOLD = 0.21133705228567123

GATE_FP_ELIGIBLE = "FP_ELIGIBLE"
GATE_ABSTAIN = "ABSTAIN"

ACTION_APPLY_LOCAL_FP = "APPLY_LOCAL_FP"
ACTION_ABSTAIN_BASELINE = "ABSTAIN_BASELINE"


SEMANTIC_ABSTENTION_CONDITIONS = (
    "empty MRI foreground",
    "empty deterministic coarse foreground",
    "no valid uncertainty-boundary hotspot candidate",
    "no valid FP correction hypothesis",
    "frozen prompt-preserving crop geometry infeasible",
)


class PolicyIntegrityError(ValueError):
    """Hard-failure condition for an invalid policy execution.

    This subclasses ValueError so the behavior remains compatible with
    the frozen M9C3 reference, where non-finite variance raised ValueError.
    """


def variance_gate(
    hotspot_variance: float,
) -> str:
    """Apply the frozen strict predictive-variance gate.

    Frozen rule:
        variance > 0.21133705228567123 -> FP_ELIGIBLE
        otherwise                     -> ABSTAIN

    Exact equality therefore abstains.

    Non-finite values are integrity failures and must never be silently
    converted into semantic abstention.
    """

    value = float(
        hotspot_variance
    )

    if not math.isfinite(
        value
    ):
        raise PolicyIntegrityError(
            "Hotspot variance must be finite."
        )

    if (
        value
        > FROZEN_VARIANCE_THRESHOLD
    ):
        return GATE_FP_ELIGIBLE

    return GATE_ABSTAIN


def final_action_for_gate(
    gate_state: str,
) -> str:
    """Map a valid frozen gate state to its final M9C3 action."""

    if gate_state == GATE_FP_ELIGIBLE:
        return ACTION_APPLY_LOCAL_FP

    if gate_state == GATE_ABSTAIN:
        return ACTION_ABSTAIN_BASELINE

    raise PolicyIntegrityError(
        f"Unknown frozen gate state: {gate_state!r}"
    )


def decide_final_action(
    hotspot_variance: float,
) -> str:
    """Return the final automatic action from hotspot variance alone."""

    return final_action_for_gate(
        variance_gate(
            hotspot_variance
        )
    )


def semantic_abstention_action(
    condition: str,
) -> str:
    """Return the frozen baseline-preserving action for valid abstention.

    Only the semantic conditions explicitly frozen in M9C3 are accepted.
    Unknown conditions are treated as implementation/integrity errors
    rather than silently becoming abstentions.
    """

    if condition not in SEMANTIC_ABSTENTION_CONDITIONS:
        raise PolicyIntegrityError(
            "Unknown semantic abstention condition: "
            f"{condition!r}"
        )

    return ACTION_ABSTAIN_BASELINE
