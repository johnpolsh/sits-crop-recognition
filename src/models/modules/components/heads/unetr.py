#

import torch
from torch import nn
from typing import Callable, Literal, Union
from ...decoders.decoder import Decoder


def intermediate_3d_conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int, int]],
        stride: Union[int, tuple[int, int, int]] = 1,
        padding: Union[int, tuple[int, int, int]] = 0,
        dilation: Union[int, tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        act_inplace: bool = True
        ) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        ),
        nn.BatchNorm3d(
            out_channels,
            eps=eps,
            momentum=momentum,
            affine=affine
            ),
        nn.ReLU(inplace=act_inplace)
    )


def intermediate_3d_deconv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int, int]],
        stride: Union[int, tuple[int, int, int]] = 1,
        padding: Union[int, tuple[int, int, int]] = 0,
        output_padding: Union[int, tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, tuple[int, int, int]] = 1,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        act_inplace: bool = True
        ) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation
            ),
        nn.BatchNorm3d(
            out_channels,
            eps=eps,
            momentum=momentum,
            affine=affine
            ),
        nn.ReLU(inplace=act_inplace)
        )


class DeconvBottleneck(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            interm_channels: list[int],
            deconv_layer: Callable[..., nn.Module] = intermediate_3d_deconv_block,
            conv_layer: Callable[..., nn.Module] = intermediate_3d_conv_block
            ):
        super().__init__()
        assert len(interm_channels) >= 1, "interm_channels should have at least 1 channel"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.interm_channels = interm_channels

        deconv_out_features = [in_channels] + interm_channels
        self.deconv = nn.Sequential(
            *(deconv_layer(
                in_channels=deconv_out_features[i],
                out_channels=deconv_out_features[i+1],
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2)
                ) for i in range(len(deconv_out_features) - 1))
            )
        
        self.conv = nn.Sequential(
            conv_layer(
                    in_channels=interm_channels[-1] * 2,
                    out_channels=interm_channels[-1],
                    kernel_size=(1, 3, 3),
                    padding=(0, 1, 1)
                    ),
            nn.ConvTranspose3d(
                in_channels=interm_channels[-1],
                out_channels=out_channels,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2)
                )
            )
        
    def forward(self, x: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = torch.cat([x, prev], dim=1)
        x = self.conv(x)
        return x


class DeconvHeadUNetTR(Decoder):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            grid_size: tuple[int, int, int],
            intermediate_out_channels: list[int] = [],
            bottleneck_layer: Callable[..., nn.Module] = DeconvBottleneck,
            voxel_reduce: Literal["none", "max", "avg", "conv"] = "conv"
            ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.intermediate_out_channels = intermediate_out_channels
        self.voxel_reduce = voxel_reduce

        out_features = intermediate_out_channels + [out_channels]
        self.deconv0 = nn.ConvTranspose3d(
            in_channels,
            out_features[0],
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
            )
        num_layers = len(out_features)
        conv_out_features = intermediate_out_channels + [out_channels]
        self.bottleneck = nn.ModuleList([
            bottleneck_layer(
                in_channels=in_channels,
                out_channels=conv_out_features[i+1],
                interm_channels=intermediate_out_channels[:i+1]
            ) for i in range(num_layers - 1)
        ])

        self.conv_reduce = nn.Identity()
        if voxel_reduce in ["conv"]:
            self.conv_reduce = nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(grid_size[0], 1, 1),
                stride=(1, 1, 1)
                )

    @property
    def decoder_params(self):
        return self.parameters()
    
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor: # type: ignore
        assert len(x) == len(self.bottleneck) + 1, f"Number of intermediates must be equal to {len(self.bottleneck)}"
        out = self.deconv0(x[-1])
        for i, inter in enumerate(reversed(x[:-1])):
            assert inter.shape[-3] == self.grid_size[0],\
                f"Expected {inter.shape[-3]=} to be equal to {self.grid_size[0]=}"
            out = self.bottleneck[i](inter, out)
        
        if self.voxel_reduce in ["avg"]:
            out = out.mean(dim=2)
        if self.voxel_reduce in ["max"]:
            out = out.max(dim=2).values
        if self.voxel_reduce in ["conv"]:
            out = self.conv_reduce(out).squeeze(2)

        return out
