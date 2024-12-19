#

import torch
from torch import nn
from torchmetrics.classification import (
    Accuracy,
    JaccardIndex
)
from torchmetrics.segmentation import (
    DiceScore,
    MeanIoU
)
from .base_module import _on_debug_hook, BaseModule


def _log_segmentation_prediction(
        seg_module: "SegmentationModule",
        y_yhat: tuple[torch.Tensor, torch.Tensor]
        ):
    y, yhat = y_yhat
    # TODO
    ...


class SegmentationModule(BaseModule):
    def __init__(
            self,
            net: nn.Module,
            on_debug_val: _on_debug_hook = _log_segmentation_prediction,
            no_default_train_metrics: bool = False,
            no_default_val_metrics: bool = False,
            no_default_test_metrics: bool = False,
            **kwargs
            ):
        super().__init__(
            net=net,
            on_debug_val=on_debug_val,
            **kwargs
            )

        if not no_default_train_metrics:
            self.train_metrics.add_module(
                "accuracy",
                Accuracy(
                    task="multiclass",
                    num_classes=net.num_classes,
                    average="weighted"
                    )
                )
            self.train_metrics.add_module(
                "jaccard",
                JaccardIndex(
                    task="multiclass",
                    num_classes=net.num_classes,
                    average="weighted"
                    )
                )
            self.train_metrics.add_module(
                "iou",
                MeanIoU(
                    num_classes=net.num_classes,
                    include_background=False,
                    input_format="one-hot"
                    )
                )
    
        if not no_default_val_metrics:
            self.val_metrics.add_module(
                "accuracy",
                Accuracy(
                    task="multiclass",
                    num_classes=net.num_classes,
                    average="weighted"
                    )
                )
            self.val_metrics.add_module(
                "jaccard",
                JaccardIndex(
                    task="multiclass",
                    num_classes=net.num_classes,
                    average="weighted"
                    )
                )
            self.val_metrics.add_module(
                "iou",
                MeanIoU(
                    num_classes=net.num_classes,
                    include_background=False,
                    input_format="one-hot"
                    )
                )
        
        if not no_default_test_metrics:
            self.test_metrics.add_module(
                "accuracy",
                Accuracy(
                    task="multiclass",
                    num_classes=net.num_classes,
                    average="weighted"
                    )
                )
            self.test_metrics.add_module(
                "jaccard",
                JaccardIndex(
                    task="multiclass",
                    num_classes=net.num_classes,
                    average="weighted"
                    )
                )
            self.test_metrics.add_module(
                "iou",
                MeanIoU(
                    num_classes=net.num_classes,
                    include_background=False,
                    input_format="one-hot"
                    )
                )
            self.test_metrics.add_module(
                "dice",
                DiceScore(
                    num_classes=net.num_classes,
                    include_background=False,
                    average="weighted",
                    input_format="one-hot"
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits
        
