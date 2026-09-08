import torch
from torch import nn


class DoubleConv3D(nn.Module):
    """Two 3D convolutions with normalization, activation, and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_probability),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class LightweightUNet3D(nn.Module):
    """Lightweight 3D U-Net for binary volumetric segmentation."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 8,
        dropout_probability: float = 0.2,
    ) -> None:
        super().__init__()

        self.encoder1 = DoubleConv3D(
            in_channels=in_channels,
            out_channels=base_channels,
            dropout_probability=dropout_probability,
        )

        self.pool1 = nn.MaxPool3d(
            kernel_size=2,
            stride=2,
        )

        self.encoder2 = DoubleConv3D(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            dropout_probability=dropout_probability,
        )

        self.pool2 = nn.MaxPool3d(
            kernel_size=2,
            stride=2,
        )

        self.bottleneck = DoubleConv3D(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            dropout_probability=dropout_probability,
        )

        self.up2 = nn.ConvTranspose3d(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            kernel_size=2,
            stride=2,
        )

        self.decoder2 = DoubleConv3D(
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            dropout_probability=dropout_probability,
        )

        self.up1 = nn.ConvTranspose3d(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            kernel_size=2,
            stride=2,
        )

        self.decoder1 = DoubleConv3D(
            in_channels=base_channels * 2,
            out_channels=base_channels,
            dropout_probability=dropout_probability,
        )

        self.output_conv = nn.Conv3d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        encoder1 = self.encoder1(x)

        encoder2 = self.encoder2(
            self.pool1(encoder1)
        )

        bottleneck = self.bottleneck(
            self.pool2(encoder2)
        )

        decoder2 = self.up2(bottleneck)

        decoder2 = torch.cat(
            (decoder2, encoder2),
            dim=1,
        )

        decoder2 = self.decoder2(decoder2)

        decoder1 = self.up1(decoder2)

        decoder1 = torch.cat(
            (decoder1, encoder1),
            dim=1,
        )

        decoder1 = self.decoder1(decoder1)

        return self.output_conv(decoder1)
