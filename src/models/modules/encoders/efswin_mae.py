#

import numpy as np
import torch
from torch import nn
from typing import (
    Callable,
    Literal
)
from .encoder import EncoderMAE
from .layers import _temporal_encoding_type
from .efswin import EFSwinTransformer
from ..decoders.swin_unet import SwinTransformerStageUp
from ..functional import (
    _int_or_tuple_2_t,
    to_2tuple
)


class SwinMAEDecoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            input_resolution: _int_or_tuple_2_t,
            depths: tuple[int, ...] = (2, 2, 6, 2),
            num_heads: tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 8,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            cat_features: bool = False,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm
            ):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = len(depths)
        self.cat_features = cat_features

        self.cat_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels // (2 ** i), in_channels // (2 ** (i + 1))),
                nn.LayerNorm(in_channels // (2 ** (i + 1))),
                nn.GELU(),
                nn.Linear(in_channels // (2 ** (i + 1)), in_channels // (2 ** (i + 1))),
                nn.Dropout(proj_drop_rate)
                ) for i in range(self.num_layers)
            ])

        input_resolution = to_2tuple(input_resolution)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        self.blocks = nn.Sequential(*[
            SwinTransformerStageUp(
                dim=int(in_channels // (2 ** i)),
                input_resolution=(
                    input_resolution[0] * (2 ** i),
                    input_resolution[1] * (2 ** i)
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

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if self.cat_features:
            x = features.pop()
            for i, feat in enumerate(reversed(features)):
                x = self.blocks[i](x)
                feat = feat
                x = torch.cat([x, feat], dim=-1)
                x = self.cat_layers[i](x)
        else:
            x = features.pop()
            x = self.blocks(x)

        x = self.norm(x)
        return x


class EFSwinMAE(EncoderMAE):
    encoder: EFSwinTransformer

    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 4,
            in_channels: int = 3,
            embed_dim: int = 96,
            num_frames: int = 3,
            tubelet_size: int = 1,
            mask_ratio: float = 0.75,
            temporal_encoding: _temporal_encoding_type = "doy",
            temporal_fusion_dropout: float = 0.1,
            depths: tuple[int, ...] = (2, 2, 6, 2),
            num_heads: tuple[int, ...] = (3, 6, 12, 24),
            decoder_depths: tuple[int, ...] = (1, 1, 2, 2),
            decoder_num_heads: tuple[int, ...] = (2, 2, 4, 6),
            cat_features: bool = False,
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
            **kwargs
            ):
        super().__init__()
        self.encoder = EFSwinTransformer( # type: ignore
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            temporal_encoding=temporal_encoding,
            temporal_fusion_dropout=temporal_fusion_dropout,
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
            weight_init="skip",
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
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.decoder = SwinMAEDecoder(
            in_channels=self.encoder.num_features,
            input_resolution=int(self.encoder.patch_embed.grid_size[1] // (2 ** self.encoder.num_layers)),
            depths=decoder_depths,
            num_heads=decoder_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            cat_features=cat_features
            )
        self.proj = nn.Linear(
            embed_dim,
            num_frames * in_channels * int(np.prod(self.encoder.patch_embed.patch_size)),
            bias=False
            )
        
        if weight_init != "skip":
            self.init_weights(weight_init)

    @torch.jit.unused
    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""]):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.encoder.init_weights(mode)

    def window_masking(
            self,
            x: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, H, W, C = x.shape
        x = x.reshape(B, T, -1, C)

        L = H * W
        # number of adjacent patches to be chosen to be masked
        k = 2 ** self.encoder.num_layers
        d_H, d_W = (k // self.encoder.patch_embed.patch_size[0], k // self.encoder.patch_embed.patch_size[1])
        p, q = (H // d_H, W // d_W)

        noise = torch.rand(B, T, p * q, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=2)
        sparse_keep = sparse_shuffle[:, :, :int(p * q * (1 - self.mask_ratio))]
        sparse_mask = sparse_shuffle[:, :, int(p * q * (1 - self.mask_ratio)):]

        indices = torch.arange(L, device=x.device).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        indices = indices.view(B, T, p, d_H, q, d_W)
        indices = indices.permute(0, 1, 2, 4, 3, 5).reshape(B, T, p * q, d_H * d_W)
        idx_keep = torch.gather(indices, dim=2, index=sparse_keep.unsqueeze(-1).expand(-1, -1, -1, d_H * d_W))
        idx_keep = idx_keep.flatten(2)
        idx_mask = torch.gather(indices, dim=2, index=sparse_mask.unsqueeze(-1).expand(-1, -1, -1, d_H * d_W))
        idx_mask = idx_mask.flatten(2)

        x_masked = x.clone()
        x_masked = torch.scatter(
            x_masked,
            dim=2,
            index=idx_mask.unsqueeze(-1).expand(-1, -1, -1, C),
            src=self.mask_token.expand(B, T, idx_mask.shape[2], C)
            )
        x_masked = x_masked.view(B, T, H, W, C)
        return x_masked, idx_keep, idx_mask

    def forward_encoder(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None
            ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        x = self.encoder.patch_embed(x)

        if temporal_coords is not None:
            x = self.encoder.add_temporal_embedding(x, temporal_coords)
        
        x, idx_keep, idx_mask = self.window_masking(x)

        x = self.encoder.fusion_layer(x)
        x = self.encoder.pos_drop(x)

        interms = []
        for i in range(self.encoder.num_layers):
            interms.append(x)
            x = self.encoder.blocks[i](x)

        x = self.encoder.norm(x)
        interms.append(x)

        return interms, idx_keep, idx_mask

    def forward_decoder(
            self,
            features: list[torch.Tensor],
            idx_keep: torch.Tensor,
            idx_mask: torch.Tensor
            ) -> torch.Tensor:
        x = self.decoder(features)
        B, H, W, _ = x.shape
        x = self.proj(x)
        x = x.view(B, H * W, self.encoder.num_frames, -1)
        x = x.permute(0, 2, 1, 3)
        return x

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        p_H, p_W = self.encoder.patch_embed.patch_size
        T = self.encoder.num_frames
        C = self.encoder.in_channels

        B = x.shape[0]
        x = x.reshape(B, C, T, x.shape[3] // p_H, p_H, x.shape[4] // p_W, p_W)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)
        H, W = x.shape[2], x.shape[3]
        x = x.reshape(B, T, H * W, -1)

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p_H, p_W = self.encoder.patch_embed.patch_size
        C = self.encoder.in_channels

        B, T = x.shape[:2]
        H = self.encoder.patch_embed.img_size[0] // p_H
        W = self.encoder.patch_embed.img_size[1] // p_W
        x = x.reshape(B, T, H, W, p_H, p_W, C)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)
        x = x.reshape(B, C, T, *self.encoder.patch_embed.img_size)

        return x

    def forward_loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = self.patchify(x)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = torch.gather(loss, dim=-1, index=mask).mean()
        return loss

    def forward(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, idx_keep, idx_mask = self.forward_encoder(x, temporal_coords)
        pred = self.forward_decoder(features, idx_keep, idx_mask)
        loss = self.forward_loss(x, pred, idx_mask)
        return pred, loss, idx_keep, idx_mask
