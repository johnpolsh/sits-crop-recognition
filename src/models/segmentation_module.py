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
from .base_module import (
    _on_debug_hook,
    BaseModule,
    log_experiment_image
)
from ..utils.plotting import mask_tensor_to_rgb_tensor


def _log_segmentation_prediction(
        seg_module: "SegmentationModule",
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        logits: torch.Tensor
        ):
    if batch_idx > 0 or seg_module.trainer.sanity_checking:
        return
    
    idx = -2
    _, y = batch
    y = y[idx].cpu().detach()
    yhat = torch.argmax(logits, dim=1)
    yhat = yhat[idx].cpu().detach()
    intersec = torch.eq(yhat, y).to(torch.long)

    y = mask_tensor_to_rgb_tensor(y, seg_module.net.num_classes)
    log_experiment_image(seg_module, y, "groud-truth")

    yhat = mask_tensor_to_rgb_tensor(yhat, seg_module.net.num_classes)
    log_experiment_image(seg_module, yhat, "prediction")

    intersec = mask_tensor_to_rgb_tensor(intersec, 2)
    log_experiment_image(seg_module, intersec, "intersection")


class SegmentationModule(BaseModule):
    def __init__(
            self,
            net: nn.Module,
            on_debug_val: _on_debug_hook = _log_segmentation_prediction,
            on_debug_test: _on_debug_hook = _log_segmentation_prediction,
            no_default_train_metrics: bool = False,
            no_default_val_metrics: bool = False,
            no_default_test_metrics: bool = False,
            **kwargs
            ):
        super().__init__(
            net=net,
            on_debug_val=on_debug_val,
            on_debug_test=on_debug_test,
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
    
        if not no_default_val_metrics:
            self.val_metrics.add_module(
                "accuracy",
                Accuracy(
                    task="multiclass",
                    num_classes=net.num_classes,
                    average="macro"
                    )
                )
            self.val_metrics.add_module(
                "jaccard",
                JaccardIndex(
                    task="multiclass",
                    num_classes=net.num_classes,
                    average="macro"
                    )
                )
            self.val_metrics.add_module(
                "iou",
                MeanIoU(
                    num_classes=net.num_classes,
                    include_background=True,
                    input_format="index"
                    )
                )
            self.val_metrics.add_module(
                "dice",
                DiceScore(
                    num_classes=net.num_classes,
                    include_background=True,
                    average="macro",
                    input_format="index"
                    )
                )
        
        if not no_default_test_metrics:
            self.test_metrics.add_module(
                "accuracy",
                Accuracy(
                    task="multiclass",
                    num_classes=net.num_classes,
                    average="macro"
                    )
                )
            self.test_metrics.add_module(
                "jaccard",
                JaccardIndex(
                    task="multiclass",
                    num_classes=net.num_classes,
                    average="macro"
                    )
                )
            self.test_metrics.add_module(
                "iou",
                MeanIoU(
                    num_classes=net.num_classes,
                    include_background=True,
                    input_format="index"
                    )
                )
            self.test_metrics.add_module(
                "dice",
                DiceScore(
                    num_classes=net.num_classes,
                    include_background=True,
                    average="macro",
                    input_format="index"
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits
