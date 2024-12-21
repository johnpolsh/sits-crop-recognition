#

import torch
from terratorch.datasets import HLSBands
from terratorch.models.backbones.prithvi_vit import (
    PRETRAINED_BANDS,
    prithvi_vit_100
)
from torch import nn
from typing import Literal, Optional, Union
from .heads.unetr import DeconvHeadUNetTR


_decoder_variants = Literal["FCNDecoder", "UNeTRDecoder"]


def _get_segmentation_head(
        variant: str,
        num_classes: int,
        input_embed_dim: int,
        **kwargs
        ) -> nn.Module:
    assert variant in ["FCNDecoder", "UNeTRDecoder"],\
        f"Got {variant=}, expected one of ['FCNDecoder', 'UNeTRDecoder']"
    
    if variant == "FCNDecoder":
        return nn.Identity()
    elif variant == "UNeTRDecoder":
        return DeconvHeadUNetTR(
            in_channels=input_embed_dim,
            out_channels=num_classes,
            intermediate_out_channels=[512, 256, 128],
            **kwargs
            )
    else:
        raise NotImplementedError(f"Unknown variant {variant=}")


class PrithviSegmentation(nn.Module):
    def __init__(
            self,
            img_size: int,
            in_chans: int,
            num_frames: int,
            num_classes: int,
            bands: list[HLSBands] = PRETRAINED_BANDS,
            weights: Optional[Literal["default"]] = "default",
            decoder: _decoder_variants = "UNeTRDecoder",
            **kwargs
            ):
        assert weights in ["default"], f"Got {weights=}, expected one of ['default']"
        assert len(bands) == in_chans, f"{bands=} must have {in_chans} elements"
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.decoder = decoder

        encoder_kwargs = {
            "bands": bands,
            "weights": weights in ["default"],
            "img_size": img_size,
            "in_chans": in_chans,
            "num_frames": num_frames,
            "features_only": True
        }

        if decoder in ["UNeTRDecoder"]:
            encoder_kwargs["out_indices"] = (2, 5, 8, 11)
        self.backbone = prithvi_vit_100(**encoder_kwargs)

        decoder_kwargs = {
            "variant": decoder,
            "num_classes": num_classes,
            "input_embed_dim": self.backbone.embed_dim,
        }
        if decoder in ["UNeTRDecoder"]:
            decoder_kwargs["grid_size"] = self.backbone.patch_embed.grid_size
            decoder_kwargs["voxel_reduce"] = kwargs.get("voxel_reduce", "conv")
        self.head = _get_segmentation_head(**decoder_kwargs)
        
    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_head(self, x: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        if self.decoder in ["UNeTRDecoder"]:
            B, _, L = x[0].shape
            F, H, W = self.backbone.patch_embed.grid_size
            x = [y[:, 1:].transpose(1, 2).view(B, L, F, H, W) for y in x]
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, F, H, W = x.shape
        assert H == W == self.img_size,\
            f"Expected {H=} and {W=} to be equal to {self.img_size=}"
        assert C == self.in_chans,\
            f"Expected {C=} to be equal to {self.in_chans=}"
        assert F == self.num_frames,\
            f"Expected {F=} to be equal to {self.num_frames=}"
        
        embeddings = self.forward_backbone(x)
        out = self.forward_head(embeddings)

        return out
