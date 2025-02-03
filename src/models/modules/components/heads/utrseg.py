#

from functools import partial
import numpy as np
import torch
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch import nn
from typing import Callable, Optional


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


class ReduceTemporalFeature(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            grid_size: tuple[int, int, int],
            pred_layer: Callable[..., nn.Module] = nn.Identity
            ):
        super().__init__()
        self.grid_size = grid_size
        self.depth_collapse = nn.Sequential(
            nn.Conv3d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=(grid_size[0], 1, 1),
                stride=1,
                padding=0
                ),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
            )
        self.pred = pred_layer(
            in_channels=embed_dim,
            out_channels=embed_dim
            )
    
    def _expand_temporal_feature(
            self,
            x: torch.Tensor
            ) -> torch.Tensor:
        B, _, N = x.shape
        x = x.transpose(1, 2)
        x = x.view(B, N, *self.grid_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = expand_temporal_feature(x, self.grid_size) # B, C, F, H, W
        x = self.depth_collapse(x) # B, C, 1, H, W
        x = x.squeeze(2) # B, C, H, W
        x = self.pred(x) # B, C, H, W
        return x


class TRSegPatchExpand(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int
            ):
        super().__init__()
        self.upconv = nn.Sequential(*[
            nn.ConvTranspose2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=2,
                stride=2
                ),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
            ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        return x

    
def _init_weights(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


class TRSegBlock(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            seq_len: int,
            num_heads: int = 1
            ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, in_features))
        self.blocks = nn.Sequential(*[
            Block(
                dim=in_features,
                num_heads=num_heads
                ) for _ in range(num_heads)
            ])
        self.patch_expand = TRSegPatchExpand(
            in_features=in_features,
            out_features=out_features
            )

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(_init_weights)
    
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = x.transpose(1, 2)
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = x.transpose(1, 2)
        x = x.view(B, C, H, W)
        x = self.patch_expand(x)
        return x


class TRDeconv(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int
            ):
        super().__init__()
        self.deconv = nn.Sequential(*[
            nn.ConvTranspose2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=2,
                stride=2
                ),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True)
            ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        return x


class TRSegHead(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int
            ):
        super().__init__()
        self.pred = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pred(x)
        return x


class UTRSegDecoder(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            grid_size: tuple[int, int, int],
            interm_out_features: list[int],
            num_layers_heads: int = 2,
            depth_collapse_layer: Callable[..., nn.Module] = ReduceTemporalFeature,
            deconv_layer: Callable[..., nn.Module] = TRDeconv,
            pred_layer: Callable[..., nn.Module] = TRSegHead,
            block_layer: Callable[..., nn.Module] = TRSegBlock
            ):
        assert in_features % grid_size[0] == 0,\
            f"Expected {in_features=} to be divisible by {grid_size[0]=}"
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.interm_out_features = interm_out_features

        block_out_features = [in_features] + interm_out_features
        num_layers = len(block_out_features) - 1
        self.clpse = nn.ModuleList([
            ReduceTemporalFeature(
                embed_dim=in_features,
                grid_size=grid_size
                ) for _ in range(num_layers)
            ])
        self.deconv = nn.ModuleList([
            nn.Sequential(*[
                TRDeconv(
                    in_features=block_out_features[j],
                    out_features=block_out_features[j+1]
                    ) for j in range(num_layers - i)
                ]) for i in range(num_layers - 1, 0, -1)
            ])

        seq_len = grid_size[1] * grid_size[2]
        self.blocks = nn.ModuleList([
            TRSegBlock(
                in_features=block_out_features[i] * 2,
                out_features=block_out_features[i+1],
                seq_len=seq_len * 4**i,
                num_heads=num_layers_heads
                ) for i in range(1, num_layers)
            ])
        self.pred = TRSegHead(
            in_features=interm_out_features[-1],
            out_features=out_features
            )
    
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        assert len(x) == len(self.blocks) + 1,\
            f"{len(x)=} must be equal to {(len(self.blocks) + 1)=}"
        out = self.clpse[0](x[-1])
        out = self.deconv[0](out)
        for i, x_i in enumerate(reversed(x[:-1])):
            x_i = self.clpse[i+1](x_i)
            x_i = self.deconv[i](x_i)
            out = torch.cat([out, x_i], dim=1)
            out = self.blocks[i](out)

        out = self.pred(out)
        return out
    