import pytest
from src.training.config import BaselineTrainingConfig


def test_baseline_training_config_has_expected_initial_settings():
    config = BaselineTrainingConfig()

    assert config.patch_size == (
        96,
        96,
        96,
    )

    assert config.batch_size == 1
    assert config.learning_rate == 1e-3

    assert config.base_channels == 8
    assert config.dropout_probability == 0.2

    assert config.positive_probability == 0.5
    assert config.validation_overlap == 0.25

    assert config.seed == 42

@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("batch_size", 0),
        ("learning_rate", 0.0),
        ("base_channels", 0),
        ("dropout_probability", -0.1),
        ("dropout_probability", 1.1),
        ("positive_probability", -0.1),
        ("positive_probability", 1.1),
        ("validation_overlap", -0.1),
        ("validation_overlap", 1.0),
        ("seed", -1),
    ],
)
def test_baseline_training_config_rejects_invalid_scalar_settings(
    field_name,
    invalid_value,
):
    with pytest.raises(ValueError):
        BaselineTrainingConfig(
            **{
                field_name: invalid_value,
            }
        )


@pytest.mark.parametrize(
    "patch_size",
    [
        (0, 96, 96),
        (96, -4, 96),
        (95, 96, 96),
        (96, 98, 96),
    ],
)
def test_baseline_training_config_rejects_invalid_patch_size(
    patch_size,
):
    with pytest.raises(ValueError):
        BaselineTrainingConfig(
            patch_size=patch_size,
        )
