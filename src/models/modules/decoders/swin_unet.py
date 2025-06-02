#

import torch
from timm.models.swin_transformer_v2 import (
    SwinTransformerV2Block
)
from torch import nn
from typing import Callable
from .decoder import (
    Decoder,
    DecoderRegistry
)
from ..functional import (
    _int_or_tuple_2_t,
    to_2tuple
)


class PatchExpand(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_scale: int = 2,
            norm_layer: Callable = nn.LayerNorm
            ):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.out_dim = dim * dim_scale
        self.norm = norm_layer(dim // dim_scale)
        self.expansion = nn.Linear(dim, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expansion(x)
        B, H, W, C = x.shape
        x = x.reshape(
            B,
            H,
            W,
            self.dim_scale,
            self.dim_scale,
            C // self.dim_scale**2
            )
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(
            B,
            H * self.dim_scale,
            W * self.dim_scale,
            C // self.dim_scale**2
            )
        x = self.norm(x)
        return x


class SwinTransformerStageUp(nn.Module):
    def __init__(
            self,
            dim: int,
            input_resolution: _int_or_tuple_2_t,
            depth: int,
            num_heads: int,
            window_size: _int_or_tuple_2_t,
            upscale: bool = True,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float | list[float] = 0.,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm
            ):
        super().__init__()

        window_size = to_2tuple(window_size)
        shift_size = (
            window_size[0] // 2,
            window_size[1] // 2
            )
        self.blocks = nn.Sequential(*[
            SwinTransformerV2Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer # type: ignore
                ) for i in range(depth)
            ])
        self.expansion = PatchExpand(
            dim,
            dim_scale=2,
            norm_layer=norm_layer
            ) if upscale else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.expansion(x)
        return x


class FinalExpandHead(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            patch_size: int,
            num_classes: int,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm
            ):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = embed_dim // patch_size

        self.expand = PatchExpand(embed_dim, patch_size, norm_layer)
        self.conv = nn.Conv2d(
            self.out_dim,
            num_classes,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


@DecoderRegistry.register("swin_unet")
class SwinUNetDecoder(Decoder):
    def __init__(
            self,
            in_channels: int,
            output_resolution: _int_or_tuple_2_t,
            scale_factor: int = 2,
            num_classes: int = 2,
            depths: tuple[int, ...] = (2, 2, 6, 2),
            num_heads: tuple[int, ...] = (3, 6, 12, 24),
            window_size: int = 8,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm
            ):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = len(depths)

        self.cat_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels // (2 ** i), in_channels // (2 ** (i + 1))),
                nn.LayerNorm(in_channels // (2 ** (i + 1))),
                nn.GELU(),
                nn.Linear(in_channels // (2 ** (i + 1)), in_channels // (2 ** (i + 1))),
                nn.Dropout(proj_drop_rate)
                ) for i in range(self.num_layers)
            ])

        output_resolution = to_2tuple(output_resolution)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        self.blocks = nn.ModuleList([
            SwinTransformerStageUp(
                dim=int(in_channels // (2 ** i)),
                input_resolution=(
                    output_resolution[0] // (2 ** (self.num_layers - i)),
                    output_resolution[1] // (2 ** (self.num_layers - i))
                    ),
                depth=depths[self.num_layers - i - 1],
                num_heads=num_heads[self.num_layers - i - 1],
                window_size=window_size,
                upscale=True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[self.num_layers - i - 1],
                norm_layer=norm_layer
                ) for i in range(self.num_layers)
            ])
        
        self.num_features = in_channels // (2 ** self.num_layers)
        self.norm = norm_layer(self.num_features)

        self.head = FinalExpandHead(
            self.num_features,
            scale_factor,
            num_classes,
            norm_layer
            )
    
    @property
    def decoder_params(self):
        return self.parameters()
    
    def _permute_feature(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        return x

    def forward(
            self,
            features: list[torch.Tensor],
            ) -> torch.Tensor:
        x = self._permute_feature(features.pop())
        for i, feat in enumerate(reversed(features)):
            x = self.blocks[i](x)
            feat = self._permute_feature(feat)
            x = torch.cat([x, feat], dim=-1)
            x = self.cat_layers[i](x)

        x = self.norm(x)
        x = self.head(x)
        return x
