#

import torch
from terratorch.datasets import HLSBands
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory
from torch import nn
from typing import Literal, Optional


class PrithviSegmentation(nn.Module):
    def __init__(
            self,
            img_size: int,
            in_chans: int,
            num_frames: int,
            num_classes: int,
            bands: list[HLSBands] = PRETRAINED_BANDS,
            weights: Optional[Literal["default"]] = "default",
            decoder: Literal["FCNDecoder"] = "FCNDecoder" # TODO: UNetDecoder
            ):
        assert weights in ["default"], f"{weights=} is not supported"
        assert len(bands) == in_chans, f"{bands=} must have {in_chans} elements"
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_frames = num_frames
        self.num_classes = num_classes

        self.backbone = EncoderDecoderFactory().build_model(
            task="segmentation",
            backbone="prithvi_vit_100",
            decoder=decoder,
            num_classes=num_classes,
            backbone_img_size=img_size,
            backbone_in_chans=in_chans,
            backbone_pretrained=bool(weights == "default"),
            backbone_num_frames=num_frames,
            backbone_bands=bands,
            necks=None,
            rescale=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.shape
        assert H == W == self.img_size,\
            f"Expected {H=} and {W=} to be equal to {self.img_size=}"
        assert C == self.in_chans * self.num_frames,\
            f"Expected {C=} to be equal to {(self.in_chans * self.num_frames)=}"
        
        x = x.view(-1, self.in_chans, self.num_frames, H, W)
        result = self.backbone(x)
        logits = result.output

        return logits
