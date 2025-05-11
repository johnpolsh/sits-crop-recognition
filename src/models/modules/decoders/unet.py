#

import torch
from torch import nn
from torch.nn import functional as F
from .decoder import (
    Decoder,
    DecoderRegistry
)
from ..functional import (
    _int_or_tuple_2_t,
    to_2tuple
)

def conv_block2d(
        in_channels: int,
        out_channels: int,
        kernel_size: _int_or_tuple_2_t,
        stride: _int_or_tuple_2_t = 1,
        padding: _int_or_tuple_2_t = 0,
        bias: bool = True
        ) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
            ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )


def deconv_block2d(
        in_channels: int,
        out_channels: int,
        kernel_size: _int_or_tuple_2_t,
        stride: _int_or_tuple_2_t = 1,
        padding: _int_or_tuple_2_t = 0,
        bias: bool = True
        ) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
            ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )


class DeconvBottleneck(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
            ):
        super().__init__()
        self.in_chans = in_channels
        self.out_chans = out_channels

        self.deconv = deconv_block2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
            )
        self.conv = nn.Sequential(
            conv_block2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
                ),
            conv_block2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
                )
            )
        
    def forward(self, x: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if x.shape[2:] != prev.shape[2:]:
            x = F.interpolate(
                x,
                size=prev.shape[2:],
                mode='bilinear',
                align_corners=True
                )
        x = torch.cat([x, prev], dim=1)
        x = self.conv(x)
        return x
    

@DecoderRegistry.register(name='unet')
class UnetDecoder(Decoder):
    def __init__(
            self,
            in_channels: int,
            output_resolution: _int_or_tuple_2_t,
            num_layers: int = 4
            ):
        super().__init__()
        self.in_channels = in_channels
        self.output_resolution = to_2tuple(output_resolution)

        out_feat = [in_channels // 2**i for i in range(num_layers + 1)]
        self.bottleneck = nn.ModuleList([
            DeconvBottleneck(
                in_channels=out_feat[i-1],
                out_channels=out_feat[i]
                ) for i in range(1, len(out_feat))
            ])
    
    @property
    def decoder_params(self):
        return self.parameters()

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = features.pop()
        for i, feat in enumerate(reversed(features)):
            x = self.bottleneck[i](x, feat)
        if x.shape[2:] != self.output_resolution:
            x = F.interpolate(
                x,
                size=self.output_resolution,
                mode='bilinear',
                align_corners=True
                )
        return x
