#

import itertools
import numpy as np
import torch
from einops import rearrange
from lightly.models import utils
from timm.layers.classifier import ClassifierHead
from timm.models.vision_transformer import get_init_weights_vit
from timm.models.swin_transformer_v2 import (
    SwinTransformerV2Block
)
from torch import nn
from torch.nn import functional as F
from typing import (
    Callable,
    Literal
)
from .encoder import EncoderMAE
from .layers import _temporal_encoding_type
from .swin import TSwinTransformer
from ..decoders.swin_unet import (
    PatchExpand,
    SwinTransformerStageUp
)
from ..functional import (
    _int_or_tuple_2_t,
    to_2tuple
)


class SwinMAEProjection(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            scale: int,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm
            ):
        super().__init__()
        self.out_dim = embed_dim // scale

        self.expand = PatchExpand(embed_dim, scale, norm_layer)
        self.proj = nn.Linear(
            self.out_dim,
            self.out_dim,
            bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.proj(x)
        return x
    

class SwinMAEDecoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_frames: int,
            output_resolution: _int_or_tuple_2_t,
            depths: tuple[int, ...] = (2, 2, 6, 2),
            num_heads: tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 8,
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
        self.num_frames = num_frames

        output_resolution = to_2tuple(output_resolution)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        self.blocks = nn.Sequential(*[
            SwinTransformerStageUp(
                dim=in_channels,
                input_resolution=(
                    output_resolution[0] // (2 ** (self.num_layers)),
                    output_resolution[1] // (2 ** (self.num_layers))
                    ),
                depth=depths[self.num_layers - i - 1],
                num_heads=num_heads[self.num_layers - i - 1],
                window_size=window_size,
                upscale=False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[self.num_layers - i - 1],
                norm_layer=norm_layer
                ) for i in range(self.num_layers)
            ])
        
        self.num_features = in_channels
        self.norm = norm_layer(self.num_features)
        self.proj = SwinMAEProjection(
            embed_dim=self.num_features,
            scale=self.num_features // out_channels * num_frames,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.norm(x)
        x = self.proj(x)
        x = x.view(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3] // self.num_frames,
            self.num_frames
            )
        x = x.permute(0, 4, 1, 2, 3)
        return x


class TSwinMAE(EncoderMAE):
    encoder: TSwinTransformer

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 4,
            in_channels: int = 3,
            num_classes: int = 2,
            embed_dim: int = 96,
            num_frames: int = 3,
            tubelet_size: int = 1,
            mask_ratio: float = 0.75,
            temporal_encoding: _temporal_encoding_type = "doy",
            depths: tuple[int, ...] = (2, 2, 6, 2),
            num_heads: tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 8,
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
            head_layer: Callable[..., nn.Module] = ClassifierHead,
            **kwargs
            ):
        super().__init__()
        self.encoder = TSwinTransformer( # type: ignore
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            temporal_encoding=temporal_encoding,
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
            global_pool=global_pool,
            weight_init=weight_init,
            head_layer=head_layer,
            **kwargs
            )
        
        self.in_channels = in_channels
        self.img_size = img_size
        self.example_input_array = torch.randn(
            1,
            in_channels,
            num_frames,
            img_size,
            img_size
            )
        
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim * self.encoder.patch_embed.grid_size[0]))

        self.decoder = SwinMAEDecoder(
            in_channels=self.encoder.num_features,
            out_channels=embed_dim,
            num_frames=num_frames,
            output_resolution=(img_size, img_size),
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer
            )

    def window_masking(
            self,
            x: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)

        L = H * W
        # number of adjacent patches to be chosen to be masked
        d_H, d_W = self.encoder.patch_embed.patch_size
        p, q = (H // d_H, W // d_W)

        noise = torch.rand(B, p * q, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_keep = sparse_shuffle[:, :int(p * q * (1 - self.mask_ratio))]
        sparse_mask = sparse_shuffle[:, int(p * q * (1 - self.mask_ratio)):]

        indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        indices = indices.view(B, p, d_H, q, d_W)
        indices = indices.permute(0, 1, 3, 2, 4).reshape(B, p * q, d_H * d_W)
        idx_keep = utils.get_at_index(indices, sparse_keep).flatten(1)
        idx_mask = utils.get_at_index(indices, sparse_mask).flatten(1)

        x_masked = x.clone()
        x_masked = torch.scatter(
            x_masked,
            dim=1,
            index=idx_mask.unsqueeze(-1).expand(-1, -1, C),
            src=self.mask_token.expand(B, idx_mask.shape[1], C)
            )
        x_masked = x_masked.view(B, H, W, C)
        return x_masked, idx_keep, idx_mask

    def forward_encoder(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder.patch_embed(x)
        print(x.shape)

        if temporal_coords is not None:
            x = self.encoder.add_temporal_embedding(x, temporal_coords)
        
        x = x.permute(0, 2, 3, 1, 4)
        x = x.flatten(3)

        x, idx_keep, idx_mask = self.window_masking(x)

        x = self.encoder.tempo_layer(x)
        x = self.encoder.pos_drop(x)

        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)

        return x, idx_keep, idx_mask

    def forward_decoder(
            self,
            x: torch.Tensor,
            idx_keep: torch.Tensor,
            idx_mask: torch.Tensor
            ) -> torch.Tensor:
        x = self.decoder(x)
        return x

    def patchify(self, pixel_values):
        patch_size_t, patch_size_h, patch_size_w = self.encoder.patch_embed.grid_size
        num_channels = self.encoder.in_channels

        # patchify
        patchified_pixel_values = rearrange(pixel_values, 'b c (t s) (h p) (w q) -> b (t h w) (s p q c)',
                                            c=num_channels, s=patch_size_t, p=patch_size_h, q=patch_size_w)

        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values):
        patch_size_t, patch_size_h, patch_size_w = self.encoder.patch_embed.grid_size
        original_height, original_width = (self.encoder.img_size,) * 2
        num_patches_h = original_height // patch_size_h
        num_patches_w = original_width // patch_size_w
        num_channels = self.encoder.in_channels

        pixel_values = rearrange(patchified_pixel_values, 'b (t h w) (s p q c) -> b c (t s) (h p) (w q)',
                                 c=num_channels, h=num_patches_h, w=num_patches_w,
                                 s=patch_size_t, p=patch_size_h, q=patch_size_w)
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        target = self.patchify(pixel_values)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss    

    def forward(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, idx_keep, idx_mask = self.forward_encoder(x, temporal_coords)
        pred = self.forward_decoder(features, idx_keep, idx_mask)
        loss = self.forward_loss(x, pred, idx_mask)
        return loss, pred, self.patchify(x), idx_keep, idx_mask
