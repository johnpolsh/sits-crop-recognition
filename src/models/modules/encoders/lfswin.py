#

import itertools
import numpy as np
import torch
from timm.layers.classifier import ClassifierHead
from timm.models.vision_transformer import get_init_weights_vit
from torch import nn
from torch.nn import functional as F
from typing import (
    Callable,
    Literal
)
from .encoder import (
    Encoder,
    EncoderDecoder
)
from .layers import (
    _temporal_encoding_type,
    PatchEmbed3D,
    TemporalEncoder
)
from .swin import SwinTransformerStage
from ..functional import (
    _int_or_tuple_2_t,
    _int_or_tuple_3_t,
    Format,
    Format3D,
    named_apply,
)
from ....utils.pylogger import RankedLogger


_logger = RankedLogger(__name__, rank_zero_only=True)


class TemporalFusionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_layers: int = 3,
            dropout: float = 0.1,
            bidirectional: bool = False,
            ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = x.view(B, T, H * W, C)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * H * W, T, C)

        x, _ = self.rnn(x)
        x = x[:, -1, :]

        x = x.view(B, H, W, -1)
        return x


class LFSwinTransformer(Encoder):
    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 4,
            in_channels: int = 3,
            num_classes: int = 2,
            embed_dim: int = 96,
            num_frames: int = 3,
            tubelet_size: int = 1,
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
        _logger.debug(f"Unused arguments `{kwargs}` in `{self.__class__.__name__}`")
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.num_classes = num_classes

        self.example_input_array = torch.randn(
            1,
            in_channels,
            num_frames,
            img_size,
            img_size
            )

        self.output_fmt = Format.NHWC
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else nn.Identity(),
            output_fmt=Format3D.NDHWC,
            bias=False
            )
        
        self.temporal_embed = TemporalEncoder(
            embed_dim=embed_dim,
            encoding_type=temporal_encoding
            )
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        grid_size = self.patch_embed.grid_size
        self.num_layers = len(depths)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        self.blocks = nn.ModuleList([
            SwinTransformerStage(
                dim=int(embed_dim * 2**i),
                input_resolution=(
                    grid_size[1] // 2**i,
                    grid_size[2] // 2**i
                    ),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downscale=True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
                ) for i in range(self.num_layers)
            ])

        self.num_features = int(embed_dim * 2 ** self.num_layers)
        self.norm = norm_layer(self.num_features)
        self.head = head_layer(
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
        head_bias = -np.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)
    
    def add_temporal_embedding(self, x, temporal_coords):
        temporal_embed = self.temporal_embed(temporal_coords)
        B, T, C = temporal_embed.shape
        temporal_embed = temporal_embed.view(B, T, 1, 1, C)
        x = x + temporal_embed
        return x

    def forward_backbone(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> torch.Tensor:
        assert x.shape[1] == self.in_channels, \
            f"Expected `{self.in_channels}` channels, got `{x.shape[1]}`"
        assert x.shape[2] == self.num_frames, \
            f"Expected `{self.num_frames}` frames, got `{x.shape[2]}`"

        x = self.patch_embed(x)

        if temporal_coords is not None:
            x = self.add_temporal_embedding(x, temporal_coords)
        
        x = self.pos_drop(x)

        for block in self.blocks:
            feats = [block(x[:, i]).unsqueeze(1) for i in range(x.shape[1])]
            x = torch.cat(feats, dim=1)

        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.head(x, pre_logits=pre_logits)
        return x

    def forward(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> torch.Tensor:
        x = self.forward_backbone(x, temporal_coords)
        x = self.forward_head(x)
        return x


class LFSwinEncoderDecoder(LFSwinTransformer, EncoderDecoder):
    def __init__(
            self,
            img_size: int = 256,
            patch_size: int = 4,
            in_channels: int = 3,
            num_classes: int = 2,
            embed_dim: int = 96,
            num_frames: int = 3,
            tubelet_size: int = 1,
            temporal_encoding: _temporal_encoding_type = "doy",
            temporal_layers: tuple[int, ...] = (3, 3, 3, 3, 3),
            temporal_dropout: float = 0.1,
            depths: tuple[int, ...] = (2, 2, 6, 2),
            num_heads: tuple[int, ...] = (3, 6, 12, 24),
            window_size: _int_or_tuple_2_t = 8,
            decoder: str = "swin_unet",
            decoder_kwargs: dict = {},
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            patch_norm: bool = True,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            weight_init: Literal["jax", "jax_nlhb", "moco", "skip", ""] = '',
            **kwargs
            ):
        super().__init__(
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
            weight_init='skip',
            **kwargs
            )
        
        self.fusion_blocks = nn.ModuleList([
            TemporalFusionBlock(
                dim=int(embed_dim * 2**i),
                num_layers=temporal_layers[i],
                dropout=temporal_dropout,
                bidirectional=False
                ) for i in range(self.num_layers + 1)
            ])

        decoder_kwargs.update({
            "in_channels": self.num_features,
            "output_resolution": self.patch_embed.grid_size[1:],
            "scale_factor": patch_size,
            "num_classes": num_classes
            })
        self.decoder = self.create_decoder(decoder, **decoder_kwargs)
        
        if weight_init != 'skip':
            self.init_weights(weight_init)

    @property
    def backbone_params(self): # TODO
        return itertools.chain(
            self.patch_embed.parameters(),
            self.pos_drop.parameters(),
            self.blocks.parameters(),
            self.norm.parameters()
            )

    def _permute_feature(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        return x

    def forward_features(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> list[torch.Tensor]:
        x = self.patch_embed(x)

        if temporal_coords is not None:
            x = self.add_temporal_embedding(x, temporal_coords)
        
        x = self.pos_drop(x)

        interms = []
        for block, fusion_block in zip(self.blocks, self.fusion_blocks):
            fused = fusion_block(x)
            interms.append(self._permute_feature(fused))
            feats = [block(x[:, i]).unsqueeze(1) for i in range(x.shape[1])]
            x = torch.cat(feats, dim=1)

        x = self.fusion_blocks[-1](x)
        x = self.norm(x)
        return interms + [self._permute_feature(x)]
    
    def forward_decoder(
            self,
            features: list[torch.Tensor]
            ) -> torch.Tensor:
        x = self.decoder(features)
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=True
            )
        return x

    def forward(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> torch.Tensor:
        features = self.forward_features(x, temporal_coords)
        x = self.forward_decoder(features)
        return x
