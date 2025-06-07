#

import torch
from terratorch.models.backbones import prithvi_mae
from torch import nn
from typing import Callable
from .encoder import EncoderMAE
from ....utils.pylogger import RankedLogger


_logger = RankedLogger(__name__, rank_zero_only=True)


class PrithviMAE(prithvi_mae.PrithviMAE, EncoderMAE):
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_channels: int = 6,
            embed_dim: int = 768,
            num_frames: int = 3,
            depth: int = 12,
            num_heads: int = 12,
            decoder_embed_dim: int = 512,
            decoder_depth: int = 8,
            decoder_num_heads: int = 16,
            mlp_ratio: float = 4.,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            norm_pix_loss: bool = False,
            coords_encoding: list[str] | None = None,
            coords_scale_learn: bool = False,
            drop_path: float = 0.,
            mask_ratio: float = 0.75,
            **kwargs
            ):
        _logger.debug(f"Unused arguments `{kwargs}` in `{self.__class__.__name__}`")
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            num_frames=num_frames,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer, # type: ignore
            norm_pix_loss=norm_pix_loss,
            coords_encoding=coords_encoding,
            coords_scale_learn=coords_scale_learn,
            drop_path=drop_path,
            mask_ratio=mask_ratio
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

    def forward_encoder(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            location_coords: torch.Tensor | None = None,
            ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        return self.encoder(x, temporal_coords, location_coords, self.mask_ratio)
    
    def forward_decoder(
            self,
            features: list[torch.Tensor],
            idx_keep: torch.Tensor,
            idx_mask: torch.Tensor,
            ) -> torch.Tensor:
        return self.decoder(features, idx_keep, idx_mask)
    
    def forward( # type: ignore
            self,
            x: torch.Tensor,
            temporal_coords: None | torch.Tensor = None,
            location_coords: None | torch.Tensor = None,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(x, temporal_coords, location_coords)
        pred = self.forward_decoder(latent, ids_restore, mask)
        loss = self.forward_loss(x, pred, mask)
        return pred, loss, ids_restore, mask
