#

import torch
from lightning.pytorch.trainer.states import RunningStage
from torch import nn
from torchmetrics.classification import (
    Accuracy,
    JaccardIndex
)
from torchmetrics.segmentation import (
    DiceScore,
    MeanIoU
)
from typing import Optional
from .base_module import _on_debug_hook, BaseModule
from ..utils.plotting import (
    get_channels_permuted_tensor,
    make_grid_tensor,
    mask_tensor_to_rgb_tensor,
    normalize_img_tensor
)


def _get_stage_name(stage: Optional[RunningStage]) -> str:
    if stage == RunningStage.TRAINING:
        return "train"
    if stage == RunningStage.SANITY_CHECKING:
        return "sanity_check"
    if stage == RunningStage.VALIDATING:
        return "val"
    if stage == RunningStage.TESTING:
        return "test"
    if stage == RunningStage.PREDICTING:
        return "predict"
    return "unknown"


def _log_segmentation_prediction(
        seg_module: "SegmentationModule",
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        logits: torch.Tensor
        ):
    if batch_idx > 0 or seg_module.trainer.sanity_checking:
        return
    
    stage = _get_stage_name(seg_module.trainer.state.stage)

    idx = 1
    x, y = batch
    x = x[idx].cpu().detach()
    x = normalize_img_tensor(x)
    x = get_channels_permuted_tensor(x, [2, 1, 0])
    x = make_grid_tensor(x, pad_value=1)
    seg_module.logger.experiment.add_image(
        f"{stage}/target",
        x,
        global_step=seg_module.current_epoch
        )

    y = y[idx].cpu().detach()
    y = mask_tensor_to_rgb_tensor(y, seg_module.net.num_classes)
    seg_module.logger.experiment.add_image(
        f"{stage}/groud_truth",
        y,
        global_step=seg_module.current_epoch
        )

    yhat = torch.argmax(logits, dim=1)
    yhat = yhat[idx].cpu().detach()
    yhat = mask_tensor_to_rgb_tensor(yhat, seg_module.net.num_classes)
    seg_module.logger.experiment.add_image(
        f"{stage}/prediction",
        yhat,
        global_step=seg_module.current_epoch
        )


class SegmentationModule(BaseModule):
    def __init__(
            self,
            net: nn.Module,
            on_debug_train: _on_debug_hook = _log_segmentation_prediction,
            on_debug_val: _on_debug_hook = _log_segmentation_prediction,
            on_debug_test: _on_debug_hook = _log_segmentation_prediction,
            no_default_train_metrics: bool = False,
            no_default_val_metrics: bool = False,
            no_default_test_metrics: bool = False,
            **kwargs
            ):
        super().__init__(
            net=net,
            on_debug_train=on_debug_train,
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
