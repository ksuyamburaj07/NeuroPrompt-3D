import torch

from src.models.unet3d import LightweightUNet3D
from src.inference.sliding_window import sliding_window_logits


def test_sliding_window_logits_preserves_full_volume_shape():
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=2,
        dropout_probability=0.0,
    )

    mri = torch.randn(
        1,
        4,
        20,
        24,
        28,
    )

    logits = sliding_window_logits(
        model=model,
        mri=mri,
        roi_size=(16, 16, 16),
        overlap=0.25,
    )

    assert logits.shape == (
        1,
        1,
        20,
        24,
        28,
    )
