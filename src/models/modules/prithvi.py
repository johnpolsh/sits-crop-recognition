#

import torch
from terratorch.datasets import HLSBands
from terratorch.models.backbones.prithvi_vit import (
    PRETRAINED_BANDS,
    prithvi_vit_100
)
from torch import nn
from typing import Literal, Optional, Union
from .components.heads.unetr import DeconvHeadUNetTR


_decoder_variants = Literal["FCNDecoder", "UNeTRDecoder", "TemporalAttention", "UTRSegDecoder"]


def _get_segmentation_head(
        variant: str,
        num_classes: int,
        input_embed_dim: int,
        **kwargs
        ) -> nn.Module:
    assert variant in ["FCNDecoder", "UNeTRDecoder", "TemporalAttention", "UTRSegDecoder"],\
        f"Got {variant=}, expected one of ['FCNDecoder', 'UNeTRDecoder']"
    
    if variant == "FCNDecoder":
        return nn.Identity()
    elif variant == "UTRSegDecoder":
        return UTRSegDecoder(
            in_features=input_embed_dim,
            out_features=num_classes,
            grid_size=kwargs.get("grid_size", (3, 8, 8)),
            interm_out_features=kwargs.get("interm_out_features", [512, 256, 128, 64]),
            num_layers_heads=kwargs.get("num_layers_heads", 1)
            )
    elif variant == "UNeTRDecoder":
        return DeconvHeadUNetTR(
            in_channels=input_embed_dim,
            out_channels=num_classes,
            intermediate_out_channels=[512, 256, 128],
            **kwargs
            )
    elif variant == "TemporalAttention":
        return TemporalAttentionDeconv(
            in_chans=input_embed_dim,
            out_chans=num_classes,
            interm_out_chans=[512, 256, 128],
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
        assert len(bands) == in_chans, f"{bands=} must have {in_chans} elements"
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.decoder = decoder

        self.example_input_array = torch.randn(
            1,
            in_chans * num_frames,
            img_size,
            img_size
        )

        encoder_kwargs = {
            "bands": bands,
            "pretrained": weights in ["default"],
            "img_size": img_size,
            "in_chans": in_chans,
            "num_frames": num_frames,
            "features_only": True
        }

        if decoder in ["UNeTRDecoder", "TemporalAttention", "UTRSegDecoder"]:
            encoder_kwargs["out_indices"] = (0, 2, 6, 11)
        self.backbone = prithvi_vit_100(**encoder_kwargs)

        decoder_kwargs = {
            "variant": decoder,
            "num_classes": num_classes,
            "input_embed_dim": self.backbone.embed_dim,
        }
        if decoder in ["UNeTRDecoder"]:
            decoder_kwargs["grid_size"] = self.backbone.patch_embed.grid_size
            decoder_kwargs["voxel_reduce"] = kwargs.get("voxel_reduce", "conv")
        if decoder in ["TemporalAttention"]:
            decoder_kwargs["grid_size"] = self.backbone.patch_embed.grid_size
        self.head = _get_segmentation_head(**decoder_kwargs)
    
    @property
    def backbone_params(self):
        return self.backbone.parameters()
    
    @property
    def head_params(self):
        return self.head.parameters()

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_head(self, x: Union[torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        if self.decoder in ["UNeTRDecoder"]:
            B, _, L = x[0].shape
            F, H, W = self.backbone.patch_embed.grid_size
            x = [y[:, 1:].transpose(1, 2).view(B, L, F, H, W) for y in x]
        if self.decoder in ["TemporalAttention"]:
            x = [y[:, 1:].transpose(1, 2) for y in x]
        if self.decoder in ["UTRSegDecoder"]:
            x = [y[:, 1:] for y in x]
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, N, H, W = x.shape

        assert H == W == self.img_size,\
            f"Expected image size {self.img_size=} but got {H=}, {W=}"
        assert N == self.in_chans * self.num_frames,\
            f"Expected concatenated channels to be {self.in_chans * self.num_frames} but got {N=}"
        
        x = x.view(-1, self.in_chans, self.num_frames, H, W)
        
        embeddings = self.forward_backbone(x)
        out = self.forward_head(embeddings)

        return out
