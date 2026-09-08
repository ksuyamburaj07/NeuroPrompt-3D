from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Configuration for the initial NeuroPrompt-3D baseline."""

    patch_size: tuple[int, int, int] = (
        96,
        96,
        96,
    )

    batch_size: int = 1
    num_epochs: int = 1
    learning_rate: float = 1e-3

    base_channels: int = 8
    dropout_probability: float = 0.2

    positive_probability: float = 0.5
    validation_overlap: float = 0.25

    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if len(self.patch_size) != 3:
            raise ValueError(
                "patch_size must contain exactly three dimensions"
            )

        if any(
            dimension <= 0
            for dimension in self.patch_size
        ):
            raise ValueError(
                "patch_size dimensions must be greater than zero"
            )

        if any(
            dimension % 4 != 0
            for dimension in self.patch_size
        ):
            raise ValueError(
                "patch_size dimensions must be divisible by 4"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be greater than zero"
            )

        if self.base_channels <= 0:
            raise ValueError(
                "base_channels must be greater than zero"
            )

        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError(
                "dropout_probability must be between 0 and 1"
            )

        if not 0.0 <= self.positive_probability <= 1.0:
            raise ValueError(
                "positive_probability must be between 0 and 1"
            )

        if not 0.0 <= self.validation_overlap < 1.0:
            raise ValueError(
                "validation_overlap must be at least 0 and less than 1"
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative"
            )

        if self.device not in {
            "auto",
             "cpu",
            "cuda",
        }:
             raise ValueError(
                "device must be one of: auto, cpu, cuda"
             )

        if self.num_epochs <= 0:
             raise ValueError(
                 "num_epochs must be greater than zero"
        )
