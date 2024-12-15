#

import torch
from lightning import LightningModule
from torch import nn, optim
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    Accuracy,
    JaccardIndex
)
from torchmetrics.segmentation import (
    DiceScore,
    MeanIoU
)
from typing import Callable, Literal, Optional, Union
from ..data.preprocessing import compute_class_weight


class BaseSegmentationModule(LightningModule):
    def __init__(
            self,
            net: nn.Module,
            criterion: Union[nn.Module, Callable[..., nn.Module]] = nn.CrossEntropyLoss(),
            optimizer: Callable[..., optim.Optimizer] = optim.Adam,
            scheduler: Optional[Callable[..., optim.lr_scheduler._LRScheduler]] = None,
            class_weight_strategy: Literal["none", "per-batch"] = "none",
            ignore_hyparams: list[str] = ["net", "criterion"]
            ):
        assert class_weight_strategy in ["none", "per-batch"],\
            f"Got {class_weight_strategy=}, expected one of ['none', 'per-batch']"

        super().__init__()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net

        assert hasattr(net, "num_classes"),\
            "Expected net to have num_classes attribute"
        if class_weight_strategy in ["per-batch"]:
            assert not isinstance(criterion, nn.Module),\
                "'per-batch' class weight strategy requires criterion to be a constructor"
        self.class_weight_strategy = class_weight_strategy

        self.save_hyperparameters(ignore=ignore_hyparams)

        self.train_loss = MeanMetric()
        self.train_jaccard = JaccardIndex(
            task="multiclass",
            num_classes=net.num_classes,
            average="weighted"
            )
        self.train_iou = MeanIoU(
            num_classes=net.num_classes,
            include_background=True,
            input_format="one-hot"
            )

        self.val_loss = MeanMetric()
        self.val_jaccard = JaccardIndex(
            task="multiclass",
            num_classes=net.num_classes,
            average="weighted"
            )
        self.val_iou = MeanIoU(
            num_classes=net.num_classes,
            include_background=True,
            input_format="one-hot"
            )

        self.test_loss = MeanMetric()
        self.test_accuracy = Accuracy(
            task="multiclass",
            num_classes=net.num_classes,
            average="weighted"
            )
        self.test_jaccard = JaccardIndex(
            task="multiclass",
            num_classes=net.num_classes,
            average="weighted"
            )
        self.test_iou = MeanIoU(
            num_classes=net.num_classes,
            include_background=True,
            input_format="one-hot"
            )
        self.test_dice = DiceScore(
            num_classes=net.num_classes,
            include_background=True,
            average="weighted",
            input_format="one-hot"
            )

    def _calculate_loss(
            self,
            logits: torch.Tensor,
            y: torch.Tensor
            ) -> torch.Tensor:
        if isinstance(self.criterion, nn.Module):
            return self.criterion(logits, y)
        
        if self.class_weight_strategy in ["per-batch"]:
            weight = compute_class_weight(y, self.net.num_classes)
            criterion = self.criterion(weight=weight)
        else:
            criterion = self.criterion()
            
        return criterion(logits, y)

    def on_train_epoch_start(self):
        self.train_loss.reset()
        self.train_jaccard.reset()
        self.train_iou.reset()

    def on_train_epoch_end(self):
        self.log("train_loss", self.train_loss.compute())
        self.log("train_jaccard", self.train_jaccard.compute())
        self.log("train_iou", self.train_iou.compute())

    def training_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
            ):
        x, y = batch
        logits = self.forward(x)

        loss = self._calculate_loss(logits, y)
        self.train_loss(loss)

        pred = logits.argmax(dim=1)
        self.train_jaccard(pred, y)
        self.train_iou(pred, y)

        return loss

    def on_validation_epoch_start(self):
        self.val_loss.reset()
        self.val_jaccard.reset()
        self.val_iou.reset()

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss.compute())
        self.log("val_jaccard", self.val_jaccard.compute())
        self.log("val_iou", self.val_iou.compute())

    def validation_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
            ):
        x, y = batch
        logits = self.forward(x)

        loss = self._calculate_loss(logits, y)
        self.val_loss(loss)

        pred = logits.argmax(dim=1)
        self.val_jaccard(pred, y)
        self.val_iou(pred, y)
    
        return loss

    def on_test_epoch_start(self):
        ...

    def on_test_epoch_end(self):
        ...

    def test_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
            ):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters())

        scheduler = []
        if self.scheduler is not None:
            config = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
                "strict": True
            }
            scheduler.append(config)
        
        return [optimizer], scheduler
