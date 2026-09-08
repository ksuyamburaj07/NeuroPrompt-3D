import torch
from torch import nn
from src.models.unet3d import DoubleConv3D, LightweightUNet3D
from src.training.losses import binary_segmentation_loss

def test_unet3d_preserves_spatial_shape_and_returns_one_output_channel():
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
    )

    mri = torch.randn(
        1,
        4,
        32,
        32,
        32,
    )

    output = model(mri)

    assert output.shape == (
        1,
        1,
        32,
        32,
        32,
    )

def test_double_conv3d_changes_channels_but_preserves_spatial_shape():
    block = DoubleConv3D(
        in_channels=4,
        out_channels=8,
        dropout_probability=0.2,
    )

    x = torch.randn(
        1,
        4,
        16,
        16,
        16,
    )

    output = block(x)

    assert output.shape == (
        1,
        8,
        16,
        16,
        16,
    )

def test_lightweight_unet3d_contains_unet_and_dropout_layers():
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
    )

    modules = list(model.modules())

    assert any(
        isinstance(module, nn.MaxPool3d)
        for module in modules
    )

    assert any(
        isinstance(module, nn.ConvTranspose3d)
        for module in modules
    )

    assert any(
        isinstance(module, nn.Dropout3d)
        for module in modules
    )

def test_lightweight_unet3d_does_not_apply_sigmoid_internally():
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
    )

    assert not any(
        isinstance(module, nn.Sigmoid)
        for module in model.modules()
    )

def test_unet3d_supports_forward_and_backward_pass():
    model = LightweightUNet3D(
        in_channels=4,
        out_channels=1,
        base_channels=4,
        dropout_probability=0.1,
    )

    mri = torch.randn(
        1,
        4,
        16,
        16,
        16,
    )

    target = torch.zeros(
        (1, 16, 16, 16),
        dtype=torch.uint8,
    )

    target[
        :,
        4:12,
        4:12,
        4:12,
    ] = 1

    logits = model(mri)

    loss = binary_segmentation_loss(
        logits,
        target,
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(
        gradient is not None
        for gradient in gradients
    )

    assert any(
        torch.any(gradient != 0)
        for gradient in gradients
    )
