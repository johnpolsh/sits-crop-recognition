#

import torch
from torch import nn
from .base_segmentation_module import BaseSegmentationModule


class SegmentationModule(BaseSegmentationModule):
    def __init__(
            self,
            net: nn.Module,
            **kwargs
            ):
        super().__init__(net=net, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits
        
