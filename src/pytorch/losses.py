#

import torch
from torch import nn
from torch.nn import functional as F
from typing import (
    Literal,
    Optional
)
from src.data.preprocessing import compute_class_weight


class DynamicWeightCrossEntropyLoss(nn.Module):
    def __init__(
            self,
            num_classes: int,
            weight_mode: Literal["per_batch", "none"] = "per_batch",
            size_average: Optional[bool] = None,
            ignore_index: int = -100,
            reduce: Optional[bool] = None,
            reduction: str = "mean",
            label_smoothing: float = 0.0,
            clip_min: float = 0.,
            clip_max: float = 10.
            ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_mode = weight_mode
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
            ) -> torch.Tensor:
        if self.weight_mode == "per_batch":
            weight = compute_class_weight(
                targets,
                self.num_classes,
                clip_min=self.clip_min,
                clip_max= self.clip_max
                )
        else:
            weight = None
        
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            size_average=self.size_average,
            ignore_index=self.ignore_index,
            reduce=self.reduce,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
            )
        
