import numpy as np
import pytest
import torch

import src.uncertainty.mc_dropout as mc_module

from src.uncertainty.hotspot import (
    HotspotIntegrityError,
    physical_boundary_shell_zyx,
    select_uncertainty_hotspot,
)

from src.uncertainty.mc_dropout import (
    EXPECTED_DROPOUT3D_MODULES,
    MC_MASTER_SEED,
    MC_PASSES,
    UncertaintyIntegrityError,
    derive_case_seed,
    disable_mc_dropout,
    enable_mc_dropout,
    probability_stack_statistics,
    run_mc_dropout_case,
    sliding_window_logits_mode_preserving,
)


class FiveDropoutToy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout1 = torch.nn.Dropout3d(0.2)
        self.dropout2 = torch.nn.Dropout3d(0.2)
        self.dropout3 = torch.nn.Dropout3d(0.2)
        self.dropout4 = torch.nn.Dropout3d(0.2)
        self.dropout5 = torch.nn.Dropout3d(0.2)

        self.identity = torch.nn.Identity()

    def forward(self, x):
        return self.identity(
            x[:, :1]
        )


def test_frozen_mc_constants():
    assert MC_PASSES == 10
    assert MC_MASTER_SEED == 20260913
    assert EXPECTED_DROPOUT3D_MODULES == 5


def test_case_seed_matches_frozen_development_records():
    assert (
        derive_case_seed(
            "BraTS-GLI-02656-100"
        )
        == 768327632
    )

    assert (
        derive_case_seed(
            "BraTS-GLI-02084-102"
        )
        == 283088994
    )

    assert (
        derive_case_seed(
            "BraTS-GLI-02334-100"
        )
        == 1552142696
    )


def test_case_seed_is_deterministic_and_case_specific():
    first = derive_case_seed(
        "synthetic-case-a"
    )

    replay = derive_case_seed(
        "synthetic-case-a"
    )

    other = derive_case_seed(
        "synthetic-case-b"
    )

    assert first == replay
    assert first != other


def test_selective_mc_dropout_activates_only_five_dropout3d_modules():
    model = FiveDropoutToy()

    activated = enable_mc_dropout(
        model
    )

    assert len(
        activated
    ) == 5

    assert model.training is False

    training_non_dropout = [
        name
        for name, module in model.named_modules()
        if (
            module.training
            and not isinstance(
                module,
                torch.nn.Dropout3d,
            )
        )
    ]

    assert training_non_dropout == []

    assert all(
        module.training
        for module in [
            model.dropout1,
            model.dropout2,
            model.dropout3,
            model.dropout4,
            model.dropout5,
        ]
    )

    disable_mc_dropout(
        model
    )

    assert all(
        not module.training
        for module in model.modules()
    )


def test_mode_preserving_sliding_window_does_not_disable_dropout():
    model = FiveDropoutToy()

    enable_mc_dropout(
        model
    )

    before = tuple(
        module.training
        for module in [
            model.dropout1,
            model.dropout2,
            model.dropout3,
            model.dropout4,
            model.dropout5,
        ]
    )

    x = torch.ones(
        (
            1,
            1,
            8,
            8,
            8,
        ),
        dtype=torch.float32,
    )

    output = (
        sliding_window_logits_mode_preserving(
            model=model,
            mri_batch=x,
            roi_size=(
                8,
                8,
                8,
            ),
            overlap=0.25,
        )
    )

    after = tuple(
        module.training
        for module in [
            model.dropout1,
            model.dropout2,
            model.dropout3,
            model.dropout4,
            model.dropout5,
        ]
    )

    assert output.shape == x.shape
    assert before == after
    assert all(after)
    assert model.training is False


def test_probability_statistics_use_population_variance():
    stack = np.stack(
        [
            np.full(
                (2, 2, 2),
                value,
                dtype=np.float32,
            )
            for value in range(
                MC_PASSES
            )
        ],
        axis=0,
    )

    mean_probability, variance = (
        probability_stack_statistics(
            stack
        )
    )

    expected_mean = np.mean(
        stack,
        axis=0,
        dtype=np.float64,
    ).astype(
        np.float32
    )

    expected_variance = np.var(
        stack,
        axis=0,
        dtype=np.float64,
        ddof=0,
    ).astype(
        np.float32
    )

    assert mean_probability.dtype == np.float32
    assert variance.dtype == np.float32

    assert np.array_equal(
        mean_probability,
        expected_mean,
    )

    assert np.array_equal(
        variance,
        expected_variance,
    )


def test_mc_case_sequence_is_replayable_and_restores_rng(monkeypatch):
    model = FiveDropoutToy()

    def fake_mode_preserving(
        model,
        mri_batch,
        roi_size,
        overlap,
    ):
        del model
        del roi_size
        del overlap

        return torch.rand(
            (
                1,
                1,
                *mri_batch.shape[-3:],
            ),
            dtype=torch.float32,
            device=mri_batch.device,
        )

    monkeypatch.setattr(
        mc_module,
        "sliding_window_logits_mode_preserving",
        fake_mode_preserving,
    )

    mri = torch.zeros(
        (
            1,
            4,
            4,
            4,
            4,
        ),
        dtype=torch.float32,
    )

    rng_before = (
        torch.get_rng_state()
        .clone()
    )

    first = run_mc_dropout_case(
        model=model,
        mri_batch=mri,
        roi_size=(
            4,
            4,
            4,
        ),
        overlap=0.25,
        case_id=(
            "BraTS-GLI-02656-100"
        ),
    )

    rng_after = (
        torch.get_rng_state()
        .clone()
    )

    replay = run_mc_dropout_case(
        model=model,
        mri_batch=mri,
        roi_size=(
            4,
            4,
            4,
        ),
        overlap=0.25,
        case_id=(
            "BraTS-GLI-02656-100"
        ),
    )

    assert torch.equal(
        rng_before,
        rng_after,
    )

    assert first.case_seed == 768327632
    assert first.pass_hashes == replay.pass_hashes

    assert np.array_equal(
        first.probability_stack,
        replay.probability_stack,
    )

    assert np.array_equal(
        first.predictive_variance,
        replay.predictive_variance,
    )

    assert len(
        set(
            first.pass_hashes
        )
    ) == 10

    assert model.training is False


def test_mc_case_requires_exactly_five_dropout_modules(monkeypatch):
    model = torch.nn.Sequential(
        torch.nn.Dropout3d(
            0.2
        )
    )

    def should_not_run(
        model,
        mri_batch,
        roi_size,
        overlap,
    ):
        raise AssertionError(
            "Inference should not be reached."
        )

    monkeypatch.setattr(
        mc_module,
        "sliding_window_logits_mode_preserving",
        should_not_run,
    )

    with pytest.raises(
        UncertaintyIntegrityError,
        match="exactly 5",
    ):
        run_mc_dropout_case(
            model=model,
            mri_batch=torch.zeros(
                (
                    1,
                    4,
                    4,
                    4,
                    4,
                )
            ),
            roi_size=(
                4,
                4,
                4,
            ),
            overlap=0.25,
            case_id="synthetic",
        )


def _base_hotspot_inputs():
    mri = np.ones(
        (
            4,
            15,
            15,
            15,
        ),
        dtype=np.float32,
    )

    coarse = np.zeros(
        (
            15,
            15,
            15,
        ),
        dtype=bool,
    )

    coarse[
        4:11,
        4:11,
        4:11,
    ] = True

    variance = np.zeros(
        coarse.shape,
        dtype=np.float32,
    )

    return (
        mri,
        coarse,
        variance,
    )


def test_hotspot_selects_maximum_variance_candidate():
    mri, coarse, variance = (
        _base_hotspot_inputs()
    )

    variance[
        4,
        7,
        7,
    ] = 0.2

    variance[
        3,
        7,
        7,
    ] = 0.7

    result = select_uncertainty_hotspot(
        mri,
        coarse,
        variance,
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    assert result.available

    assert result.coordinate_zyx == (
        3,
        7,
        7,
    )

    assert np.isclose(
        result.predictive_variance,
        0.7,
    )

    assert result.abstention_condition is None


def test_hotspot_tie_uses_c_order_first_coordinate():
    mri, coarse, variance = (
        _base_hotspot_inputs()
    )

    first = (
        3,
        6,
        6,
    )

    second = (
        3,
        6,
        7,
    )

    variance[
        first
    ] = 0.8

    variance[
        second
    ] = 0.8

    result = select_uncertainty_hotspot(
        mri,
        coarse,
        variance,
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    assert result.coordinate_zyx == first


def test_hotspot_requires_mri_foreground():
    mri, coarse, variance = (
        _base_hotspot_inputs()
    )

    mri[...] = 0

    result = select_uncertainty_hotspot(
        mri,
        coarse,
        variance,
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    assert not result.available

    assert (
        result.abstention_condition
        == "empty MRI foreground"
    )


def test_hotspot_requires_nonempty_coarse_foreground():
    mri, coarse, variance = (
        _base_hotspot_inputs()
    )

    coarse[...] = False

    result = select_uncertainty_hotspot(
        mri,
        coarse,
        variance,
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    assert not result.available

    assert (
        result.abstention_condition
        == "empty deterministic coarse foreground"
    )


def test_hotspot_requires_candidate_overlap_with_mri_foreground():
    mri = np.zeros(
        (
            4,
            40,
            40,
            40,
        ),
        dtype=np.float32,
    )

    # MRI foreground is far from the coarse boundary.
    mri[
        :,
        0:2,
        0:2,
        0:2,
    ] = 1.0

    coarse = np.zeros(
        (
            40,
            40,
            40,
        ),
        dtype=bool,
    )

    coarse[
        25:35,
        25:35,
        25:35,
    ] = True

    variance = np.zeros(
        coarse.shape,
        dtype=np.float32,
    )

    result = select_uncertainty_hotspot(
        mri,
        coarse,
        variance,
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    assert not result.available

    assert (
        result.abstention_condition
        == "no valid uncertainty-boundary hotspot candidate"
    )


def test_hotspot_mri_foreground_restricts_selection():
    mri, coarse, variance = (
        _base_hotspot_inputs()
    )

    mri[...] = 0

    allowed = (
        4,
        7,
        7,
    )

    forbidden = (
        3,
        7,
        7,
    )

    mri[
        :,
        allowed[0],
        allowed[1],
        allowed[2],
    ] = 1.0

    variance[
        allowed
    ] = 0.4

    variance[
        forbidden
    ] = 0.9

    result = select_uncertainty_hotspot(
        mri,
        coarse,
        variance,
        spacing_zyx_mm=(
            1.0,
            1.0,
            1.0,
        ),
    )

    assert result.coordinate_zyx == allowed


def test_hotspot_boundary_shell_uses_physical_spacing():
    coarse = np.zeros(
        (
            9,
            9,
            9,
        ),
        dtype=bool,
    )

    coarse[
        2:7,
        2:7,
        2:7,
    ] = True

    shell = physical_boundary_shell_zyx(
        coarse,
        spacing_zyx_mm=(
            5.0,
            2.0,
            1.0,
        ),
        shell_mm=2.0,
    )

    # One Z voxel outside is physically 5 mm away.
    assert not shell[
        1,
        4,
        4,
    ]

    # One Y voxel outside is exactly 2 mm away.
    assert shell[
        4,
        1,
        4,
    ]

    # One X voxel outside is 1 mm away.
    assert shell[
        4,
        4,
        1,
    ]


def test_nonfinite_variance_is_integrity_failure():
    mri, coarse, variance = (
        _base_hotspot_inputs()
    )

    variance[
        0,
        0,
        0,
    ] = np.nan

    with pytest.raises(
        HotspotIntegrityError,
        match="variance contains non-finite",
    ):
        select_uncertainty_hotspot(
            mri,
            coarse,
            variance,
            spacing_zyx_mm=(
                1.0,
                1.0,
                1.0,
            ),
        )
