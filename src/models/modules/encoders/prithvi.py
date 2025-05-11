# TODO

from terratorch.datasets import HLSBands
from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from .encoder import (
    Encoder,
    EncoderDecoder
)


class PrithviEncoder(Encoder):
    def __init__(
            self,
            img_size: int,
            in_channels: int,
            num_classes: int,
            num_frames: int,
            tubelet_size: int,
            bands: list[HLSBands] = PRETRAINED_BANDS,
    )
