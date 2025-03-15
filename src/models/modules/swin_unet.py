#

import itertools
import math
import torch
from timm.layers import ClassifierHead
from timm.models.vision_transformer import get_init_weights_vit
from timm.models.swin_transformer_v2 import (
    PatchEmbed,
    SwinTransformerV2Block
)
from torch import nn
from typing import Callable, Literal, Optional, Union
from ..functional import named_apply


class PatchMerging(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_scale: int = 2,
            norm_layer: Callable = nn.LayerNorm
            ):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.out_dim = dim_scale * dim
        self.norm = norm_layer(dim * dim_scale**2)
        self.reduction = nn.Linear(dim * dim_scale**2, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert H % self.dim_scale == 0,\
            f"Expected height `{H}` to be divisible by `{self.dim_scale}`"
        assert W % self.dim_scale == 0,\
            f"Expected width `{W}` to be divisible by `{self.dim_scale}`"
        x = x.reshape(
            B,
            H // self.dim_scale,
            self.dim_scale,
            W // self.dim_scale,
            self.dim_scale,
            C
            )
        x = x.permute(0, 1, 3, 2, 4, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x
    

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


class SwinTransformerV2Stage(nn.Module):
    def __init__(
            self,
            dim: int,
            input_resolution: Union[int, tuple[int, int]],
            depth: int,
            num_heads: int,
            window_size: int,
            downscale: bool = True,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[float, list[float]] = 0.,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm
            ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            SwinTransformerV2Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer # type: ignore
                ) for i in range(depth)
            ])
        self.merging = PatchMerging(
            dim,
            dim_scale=2,
            norm_layer=norm_layer
            ) if downscale else nn.Identity()

    def forward(
            self,
            x: torch.Tensor
            ) -> torch.Tensor:
        x = self.blocks(x)
        x = self.merging(x)
        return x


class SwinTransformerV2(nn.Module):
    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 4,
            in_channels: int = 3,
            num_classes: int = 2,
            embed_dim: int = 96,
            depths: tuple[int, ...] = (2, 2, 6, 2),
            num_heads: tuple[int, ...] = (3, 6, 12, 24),
            window_size: int = 8,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            patch_norm: bool = True,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            global_pool: Literal["", "avg"] = 'avg',
            weight_init: Literal["jax", "jax_nlhb", "moco", "skip", ""] = '',
            **kwargs
            ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.example_input_array = torch.randn(
            1,
            in_channels,
            img_size,
            img_size
            )

        self.output_fmt = "NHWC"
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            output_fmt=self.output_fmt
            )
        grid_size = self.patch_embed.grid_size
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_layers = len(depths)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        self.blocks = nn.Sequential(*[
            SwinTransformerV2Stage(
                dim=int(embed_dim * 2**i),
                input_resolution=(
                    grid_size[0] // 2**i, # type: ignore
                    grid_size[1] // 2**i # type: ignore
                    ),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downscale=i < self.num_layers - 1,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
                ) for i in range(self.num_layers)
            ])
        
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        self.norm = norm_layer(self.num_features)
        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt
            )
        if weight_init != 'skip':
            self.init_weights(weight_init)

    @property
    def backbone_params(self):
        return itertools.chain(
            self.patch_embed.parameters(),
            self.pos_drop.parameters(),
            self.blocks.parameters(),
            self.norm.parameters()
            )

    @property
    def head_params(self):
        return self.head.parameters()

    @torch.jit.unused
    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""]):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)
    
    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.head(x, pre_logits=pre_logits)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_backbone(x)
        x = self.forward_head(x)
        return x


class SwinTransformerV2StageUp(nn.Module):
    def __init__(
            self,
            dim: int,
            input_resolution: Union[int, tuple[int, int]],
            depth: int,
            num_heads: int,
            window_size: int,
            upscale: bool = True,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[float, list[float]] = 0.,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm
            ):
        super().__init__()
        self.cat = nn.Linear(
            2 * dim,
            dim
            )
        self.blocks = nn.Sequential(*[
            SwinTransformerV2Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
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

    def forward(
            self,
            x: torch.Tensor,
            skip: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
        if skip is not None:
            x = torch.cat([x, skip], dim=-1)
            x = self.cat(x)
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
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


class SwinV2Unet(SwinTransformerV2):
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 4,
            in_channels: int = 3,
            num_classes: int = 2,
            embed_dim: int = 96,
            depths: tuple[int, ...] = (2, 4, 6, 4),
            num_heads: tuple[int, ...] = (3, 8, 12, 24),
            window_size: int = 8,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            patch_norm: bool = True,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            weight_init: Literal["jax", "jax_nlhb", "moco", "skip", ""] = '',
            head_layer: Callable[..., nn.Module] = FinalExpandHead,
            **kwargs
            ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            norm_layer=norm_layer,
            weight_init='skip',
            **kwargs
            )

        grid_size = self.patch_embed.grid_size
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        self.up_blocks = nn.ModuleList([
            SwinTransformerV2StageUp(
                dim=int(embed_dim * 2 ** (self.num_layers - i - 1)),
                input_resolution=(
                    grid_size[0] // 2 ** (self.num_layers - i - 1), # type: ignore
                    grid_size[1] // 2 ** (self.num_layers - i - 1) # type: ignore
                    ),
                depth=depths[self.num_layers - i - 1],
                num_heads=num_heads[self.num_layers - i - 1],
                window_size=window_size,
                upscale=i < self.num_layers - 1,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[self.num_layers - i - 1],
                norm_layer=norm_layer
                ) for i in range(self.num_layers)
            ])
        
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim)
        self.head = head_layer(embed_dim, patch_size, num_classes)

        if weight_init != 'skip':
            self.init_weights(weight_init)

    @property
    def head_params(self):
        return itertools.chain(
            self.up_blocks.parameters(),
            self.norm_up.parameters(),
            self.head.parameters()
            )

    def forward_down(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        interms = []
        for i in range(self.num_layers):
            interms.append(x)
            x = self.blocks[i](x)
        x = self.norm(x)
        return x, interms

    def forward_up(self, x: torch.Tensor, interms: list[torch.Tensor]) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.up_blocks[i](x, interms[self.num_layers - i - 1])
        x = self.norm_up(x)
        x = self.head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, interm = self.forward_down(x)
        x = self.forward_up(x, interm)
        return x
    