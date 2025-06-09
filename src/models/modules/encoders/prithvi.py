#

import itertools
import torch
import torch.nn.functional as F
from terratorch.models.backbones.prithvi_vit import PrithviViT
from timm.layers.classifier import ClassifierHead
from torch import nn
from typing import Callable, Literal
from .encoder import (
    Encoder,
    EncoderDecoder
)
from ..components.heads.unetr import DeconvHeadUNetTR
from ..functional import (
    _int_or_tuple_2_t,
    _int_or_tuple_3_t,
    to_2tuple,
    to_3tuple,
    Format
)
from ....utils.pylogger import RankedLogger


_logger = RankedLogger(__name__, rank_zero_only=True)


class Prithvi(PrithviViT, Encoder):
    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: _int_or_tuple_3_t = (1, 16, 16),
            in_channels: int = 6,
            num_classes: int = 2,
            embed_dim: int = 768,
            num_frames: int = 3,
            depth: int = 24,
            num_heads: int = 16,
            mlp_ratio: float = 4.,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            coords_encoding: list[str] | None = None,
            coords_scale_learn: bool = False,
            drop_path: float = 0.,
            drop_rate: float = 0.,
            global_pool: Literal["", "avg"] = 'avg',
            head_layer: Callable[..., nn.Module] = ClassifierHead,
            **kwargs
            ):
        _logger.debug(f"Unused arguments `{kwargs}` in `{self.__class__.__name__}`")
        super().__init__(
            img_size=img_size,
            patch_size=patch_size, # type: ignore
            in_chans=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer, # type: ignore
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
            drop_path=drop_path,
            )

        self.img_size = to_2tuple(img_size)
        self.example_input_array = torch.randn(
            1,
            in_channels,
            num_frames,
            *self.img_size,
            )
        self.output_fmt = Format.NHWC
        self.head = head_layer(
            embed_dim * self.patch_embed.grid_size[0],
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt
            )

    @property
    def backbone_params(self):
        return itertools.chain(
            self.patch_embed.parameters(),
            *[block.parameters() for block in self.blocks]
            )

    @property
    def head_params(self):
        return self.head.parameters()

    def forward_backbone(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            location_coords: torch.Tensor | None = None
            ) -> torch.Tensor:
        return self.forward_features(x, temporal_coords, location_coords)[-1]

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    def forward_features(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            location_coords: torch.Tensor | None = None
            ) -> list[torch.Tensor]:
        features = super().forward_features(x, temporal_coords, location_coords)
        B, _, L = features[0].shape
        T, H, W = self.patch_embed.grid_size
        features = [y[:, 1:].reshape(B, H, W, L * T) for y in features]
        return features

    def forward( # type: ignore
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            location_coords: torch.Tensor | None = None
            ) -> torch.Tensor:
        x = self.forward_backbone(x, temporal_coords, location_coords)
        x = self.forward_head(x)
        return x


class PrithviEncoderDecoder(EncoderDecoder):
    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: _int_or_tuple_3_t = (1, 16, 16),
            in_channels: int = 6,
            num_classes: int = 2,
            embed_dim: int = 768,
            num_frames: int = 3,
            depth: int = 24,
            num_heads: int = 16,
            mlp_ratio: float = 4.,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            coords_encoding: list[str] | None = None,
            coords_scale_learn: bool = False,
            drop_path: float = 0.1,
            drop_rate: float = 0.,
            global_pool: Literal["", "avg"] = 'avg',
            head_layer: Callable[..., nn.Module] = ClassifierHead,
            **kwargs
            ):
        super().__init__()
        self.encoder = Prithvi(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_frames=num_frames,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
            drop_path=drop_path,
            drop_rate=drop_rate,
            global_pool=global_pool,
            head_layer=head_layer,
            **kwargs
            )
        
        self.decoder = DeconvHeadUNetTR(
            in_channels=embed_dim,
            out_channels=num_classes,
            grid_size=to_3tuple(self.encoder.patch_embed.grid_size),
            intermediate_out_channels=[512, 256, 128, 64]
            )
        
        self.example_input_array = self.encoder.example_input_array
        self.num_classes = num_classes

    @property
    def backbone_params(self):
        return self.encoder.backbone_params

    @property
    def head_params(self):
        return self.decoder.parameters()

    def forward_decoder(
            self,
            features: list[torch.Tensor]
            ) -> torch.Tensor:
        x = self.decoder(features)
        x = F.interpolate(
            x,
            size=(self.encoder.img_size[0], self.encoder.img_size[1]), # type: ignore
            mode='bilinear',
            align_corners=True
            )
        return x
    
    def forward(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> torch.Tensor:
        features = self.encoder.forward_features(x, temporal_coords)
        T, H, W = self.encoder.patch_embed.grid_size # type: ignore
        features = [feat.permute(0, 3, 1, 2).reshape(x.shape[0], -1, T, H, W) for i, feat in enumerate(features) if i in (0, 1, 3, 6, 11)]
        x = self.forward_decoder(features)
        return x
