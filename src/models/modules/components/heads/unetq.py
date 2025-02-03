#

import torch
from torch import nn
from typing import Callable, Optional, Union


def intermediate_2d_conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        conv_layer: Callable[..., nn.Module] = nn.Conv2d,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        act_layer: Callable[..., nn.Module] = nn.ReLU,
        ) -> nn.Module:
    return nn.Sequential(
        conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
            ),
        norm_layer(out_channels),
        act_layer(inplace=True)
        )


def intermediate_2d_deconv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        conv_layer: Callable[..., nn.Module] = nn.ConvTranspose2d,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        act_layer: Callable[..., nn.Module] = nn.ReLU,
        ) -> nn.Module:
    return nn.Sequential(
        conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
            ),
        norm_layer(out_channels),
        act_layer(inplace=True)
        )


def expand_temporal_feature(
        x: torch.Tensor,
        grid_size: tuple[int, ...],
        embed_dim: Optional[int] = None
        ) -> torch.Tensor:
    B, N, L = x.shape
    seq_len = grid_size[0] * grid_size[1] * grid_size[2]
    assert N == seq_len, f"{N=} must be equal to {seq_len=}"
    if embed_dim is not None:
        assert L == embed_dim, f"{L=} must be equal to {embed_dim=}"
    
    x = x.transpose(1, 2)
    x = x.reshape(B, L, *grid_size)
    return x


class SelfAttentionLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 1,
            dropout: float = 0.1
            ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        return x


class MergeLayer(nn.Module):
    def __init__(
            self,
            in_chans: int,
            grid_size: tuple[int, int, int]
            ):
        super().__init__()
        self.grid_size = grid_size
        self.seq_len = grid_size[0] * grid_size[1] * grid_size[2]
        self.merge = nn.Sequential(
            nn.Conv3d(
                in_channels=in_chans,
                out_channels=in_chans,
                kernel_size=(grid_size[0], 1, 1),
                stride=1,
                padding=0
                ),
            nn.BatchNorm3d(in_chans),
            nn.ReLU(inplace=True)
            )
        self.pred = nn.Conv2d(
            in_channels=in_chans,
            out_channels=in_chans,
            kernel_size=3,
            stride=1,
            padding=1
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        B, N, _ = x.shape
        x = x.reshape(B, N, *self.grid_size)
        x = self.merge(x)
        x = x.squeeze(2)
        x = self.pred(x)
        return x


class MergeAttentionBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            grid_size: tuple[int, int, int],
            attn_layer: Callable[..., nn.Module] = SelfAttentionLayer,
            merge_layer: Callable[..., nn.Module] = MergeLayer
            ):
        assert len(grid_size) == 3, f"{grid_size=} must have 3 elements"
        super().__init__()
        self.grid_size = grid_size
        self.seq_len = grid_size[0] * grid_size[1] * grid_size[2]
        self.attn = attn_layer(embed_dim=embed_dim)
        self.merge = merge_layer(
            in_chans=embed_dim,
            grid_size=grid_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_att = self.attn(x)
        x = x.transpose(1, 2)
        # x = torch.cat([x, x_att], dim=2)
        x = self.merge(x)
        return x
    

class DeconvBottleneck(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            interm_out_chans: list[int],
            deconv_layer: Callable[..., nn.Module] = intermediate_2d_deconv_block,
            conv_layer: Callable[..., nn.Module] = intermediate_2d_conv_block
            ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.intermediate_out_channels = interm_out_chans

        out_features = [in_chans] + interm_out_chans
        self.deconv = nn.Sequential(*[
            deconv_layer(
                in_channels=out_features[i],
                out_channels=out_features[i+1],
                kernel_size=2,
                stride=2
                ) for i in range(len(out_features) - 1)
            ])
        self.conv = nn.Sequential(
            conv_layer(
                in_channels=interm_out_chans[-1] * 2,
                out_channels=interm_out_chans[-1],
                kernel_size=3,
                padding=1
                ),
            nn.ConvTranspose2d(
                in_channels=interm_out_chans[-1],
                out_channels=out_chans,
                kernel_size=2,
                stride=2
                )
            )
        
    def forward(self, x: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = torch.cat([x, prev], dim=1)
        x = self.conv(x)
        return x


class TemporalAttentionDeconv(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            grid_size: tuple[int, int, int],
            interm_out_chans: list[int] = [],
            bottleneck_layer: Callable[..., nn.Module] = DeconvBottleneck,
            merge_attn_layer: Callable[..., nn.Module] = MergeAttentionBlock
            ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.grid_size = grid_size
        self.interm_out_chans = interm_out_chans

        out_features = interm_out_chans + [out_chans]
        self.merge_attn = merge_attn_layer(
            embed_dim=in_chans,
            grid_size=grid_size
            )
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_chans,
            out_channels=out_features[0],
            kernel_size=2,
            stride=2
            )

        num_layers = len(out_features)
        self.bottleneck = nn.ModuleList([
            bottleneck_layer(
                in_chans=in_chans,
                out_chans=out_features[i+1],
                interm_out_chans=interm_out_chans[:i+1]
                ) for i in range(num_layers - 1)
            ])
        
        self.merge = nn.ModuleList([
            merge_attn_layer(
                embed_dim=in_chans,
                grid_size=grid_size
                ) for _ in range(num_layers - 1)
            ])
        
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        out = self.merge_attn(x[-1])
        out = self.deconv(out)
        for i, x_i in enumerate(reversed(x[:-1])):
            x_i = self.merge[i](x_i)
            out = self.bottleneck[i](x_i, out)
        
        return out
