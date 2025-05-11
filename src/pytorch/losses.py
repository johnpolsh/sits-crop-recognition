#

import torch
from torch import nn
from torch.nn import functional as F
from typing import (
    Callable,
    Literal,
    Optional
)
from segmentation_models_pytorch.losses import dice, focal
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
            clip_max: float = 10.,
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
                ignore_index=self.ignore_index,
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
        

class HausdorffLoss(nn.Module):
    def __init__(
            self,
            ignore_index: int = -100,
            reduction: str = "mean",
            ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
            ) -> torch.Tensor:
        # Implement Hausdorff loss calculation here
        pass


class CompositeLoss(nn.Module):
    def __init__(
            self,
            criterions: list[Callable[..., torch.Tensor]],
            weights: Optional[list[float]] = None
            ):
        super().__init__()
        self.criterions = criterions
        self.weights = weights if weights is not None else [1.0] * len(criterions)
        assert len(self.criterions) == len(self.weights), "Length of criterions and weights must be equal"
        assert all(isinstance(w, (int, float)) for w in self.weights), "All weights must be int or float"
    
    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
            ) -> torch.Tensor:
        losses = []
        for criterion, weight in zip(self.criterions, self.weights):
            losses.append(weight * criterion(logits, targets))
        loss = torch.stack(losses).sum()
        return loss


class CEDLoss(CompositeLoss):
    def __init__(
            self,
            ignore_index: int = -100,
            weights: list[float] = [0.5, 0.5],
            ):
        super().__init__([
            nn.CrossEntropyLoss(ignore_index=ignore_index),
            dice.DiceLoss(
                mode=dice.MULTICLASS_MODE,
                from_logits=True,
                ignore_index=ignore_index
                )
            ],
            weights=weights
            )
        self.ignore_index = ignore_index
        self.mode = dice.MULTICLASS_MODE


class DCEDLoss(CompositeLoss):
    def __init__(
            self,
            num_classes: int = 19,
            ignore_index: int = -100,
            weights: list[float] = [0.5, 0.5],
            ):
        super().__init__([
            DynamicWeightCrossEntropyLoss(
                num_classes=num_classes,
                ignore_index=ignore_index
                ),
            dice.DiceLoss(
                mode=dice.MULTICLASS_MODE,
                from_logits=True,
                ignore_index=ignore_index
                )
            ],
            weights=weights
            )
        self.ignore_index = ignore_index
        self.mode = dice.MULTICLASS_MODE


class CEDFLoss(CompositeLoss):
    def __init__(
            self,
            ignore_index: int = -100,
            weights: list[float] = [0.5, 0.5, 0.2],
            ):
        super().__init__([
            nn.CrossEntropyLoss(ignore_index=ignore_index),
            dice.DiceLoss(
                mode=dice.MULTICLASS_MODE,
                from_logits=True,
                ignore_index=ignore_index
                ),
            focal.FocalLoss(
                mode=dice.MULTICLASS_MODE,
                ignore_index=ignore_index
                )
            ],
            weights=weights
            )
        self.ignore_index = ignore_index
        self.mode = dice.MULTICLASS_MODE
