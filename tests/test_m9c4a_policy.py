import numpy as np
import pytest

from src.pipeline.policy import (
    ACTION_ABSTAIN_BASELINE,
    ACTION_APPLY_LOCAL_FP,
    FROZEN_VARIANCE_THRESHOLD,
    GATE_ABSTAIN,
    GATE_FP_ELIGIBLE,
    PolicyIntegrityError,
    SEMANTIC_ABSTENTION_CONDITIONS,
    decide_final_action,
    final_action_for_gate,
    semantic_abstention_action,
    variance_gate,
)


def test_variance_gate_below_threshold_abstains():
    value = float(
        np.nextafter(
            FROZEN_VARIANCE_THRESHOLD,
            -np.inf,
        )
    )

    assert variance_gate(value) == GATE_ABSTAIN


def test_variance_gate_exact_threshold_abstains():
    assert (
        variance_gate(
            FROZEN_VARIANCE_THRESHOLD
        )
        == GATE_ABSTAIN
    )


def test_variance_gate_above_threshold_is_fp_eligible():
    value = float(
        np.nextafter(
            FROZEN_VARIANCE_THRESHOLD,
            np.inf,
        )
    )

    assert (
        variance_gate(value)
        == GATE_FP_ELIGIBLE
    )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_variance_gate_nonfinite_is_hard_failure(value):
    with pytest.raises(
        PolicyIntegrityError,
        match="must be finite",
    ):
        variance_gate(value)


def test_frozen_gate_states_map_to_exact_final_actions():
    assert (
        final_action_for_gate(
            GATE_FP_ELIGIBLE
        )
        == ACTION_APPLY_LOCAL_FP
    )

    assert (
        final_action_for_gate(
            GATE_ABSTAIN
        )
        == ACTION_ABSTAIN_BASELINE
    )


def test_unknown_gate_state_is_integrity_failure():
    with pytest.raises(
        PolicyIntegrityError,
        match="Unknown frozen gate state",
    ):
        final_action_for_gate(
            "FN_ELIGIBLE"
        )


def test_complete_decision_function():
    assert (
        decide_final_action(
            0.229637548
        )
        == ACTION_APPLY_LOCAL_FP
    )

    assert (
        decide_final_action(
            0.117870085
        )
        == ACTION_ABSTAIN_BASELINE
    )


@pytest.mark.parametrize(
    "condition",
    SEMANTIC_ABSTENTION_CONDITIONS,
)
def test_every_frozen_semantic_condition_returns_baseline(condition):
    assert (
        semantic_abstention_action(
            condition
        )
        == ACTION_ABSTAIN_BASELINE
    )


def test_unknown_semantic_condition_is_not_silently_abstained():
    with pytest.raises(
        PolicyIntegrityError,
        match="Unknown semantic abstention condition",
    ):
        semantic_abstention_action(
            "unexpected tensor geometry"
        )


def test_frozen_threshold_has_not_been_rounded():
    assert (
        FROZEN_VARIANCE_THRESHOLD
        == 0.21133705228567123
    )
