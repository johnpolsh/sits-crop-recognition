#

import torch
from lightning import LightningModule
from pathlib import Path
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torchmetrics import MeanMetric
from typing import Any, Callable, Literal, Optional, TypedDict, Union
from ..data.preprocessing import compute_class_weight


_on_debug_hook = Callable[[Any, tuple[torch.Tensor, ...], int, torch.Tensor], None]


class MetricParam(TypedDict):
    alias: Optional[str]
    metric: Callable[..., nn.Module]


class BaseModule(LightningModule):
    def __init__(
            self,
            net: nn.Module,
            criterion: Union[_Loss, Callable[..., _Loss]] = nn.CrossEntropyLoss(),
            optimizer: Callable[..., optim.Optimizer] = optim.Adam,
            scheduler: Optional[Callable[..., optim.lr_scheduler._LRScheduler]] = None,
            train_metrics: list[MetricParam] = [],
            val_metrics: list[MetricParam] = [],
            test_metrics: list[MetricParam] = [],
            on_debug_hook: Optional[_on_debug_hook] = None,
            on_debug_val: Optional[_on_debug_hook] = None,
            on_debug_test: Optional[_on_debug_hook] = None,
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
        self.on_debug_hook = on_debug_hook
        self.on_debug_val = on_debug_val
        self.on_debug_test = on_debug_test

        torch.set_float32_matmul_precision("medium")

        assert hasattr(net, "num_classes"),\
            "Expected net to have num_classes attribute"
        self.class_weight_strategy = class_weight_strategy

        self.save_hyperparameters(ignore=ignore_hyparams)

        self.train_loss = MeanMetric()
        self.train_metrics = nn.ModuleDict({
            (param["alias"] or param["metric"].__name__): param["metric"]()
            for param in train_metrics
        })

        self.val_loss = MeanMetric()
        self.val_metrics = nn.ModuleDict({
            (param["alias"] or param["metric"].__name__): param["metric"]()
            for param in val_metrics
        })

        self.test_loss = MeanMetric()
        self.test_metrics = nn.ModuleDict({
            (param["alias"] or param["metric"].__name__): param["metric"]()
            for param in test_metrics
        })

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
        for metric in self.train_metrics.values():
            metric.reset()

    def on_train_epoch_end(self):
        self.log("train/loss", self.train_loss.compute(), prog_bar=True)
        for name, metric in self.train_metrics.items():
            self.log(f"train/{name}", metric.compute(), prog_bar=True)

    def training_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
            ):
        x, y = batch
        logits = self.forward(x)

        loss = self._calculate_loss(logits, y)
        self.train_loss.update(loss)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        pred = torch.argmax(logits, dim=1)
        for metric in self.train_metrics.values():
            metric.update(pred, y)

        if self.on_debug_hook is not None:
            self.on_debug_hook(self, batch, batch_idx, logits)

        return loss

    def on_validation_epoch_start(self):
        self.val_loss.reset()
        for metric in self.val_metrics.values():
            metric.reset()

    def on_validation_epoch_end(self):
        self.log("val/loss", self.val_loss.compute(), prog_bar=True)
        for name, metric in self.val_metrics.items():
            self.log(f"val/{name}", metric.compute(), prog_bar=True)

    def validation_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
            ):
        x, y = batch
        logits = self.forward(x)

        loss = self._calculate_loss(logits, y)
        self.val_loss.update(loss)
        self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        pred = torch.argmax(logits, dim=1)
        for metric in self.val_metrics.values():
            metric.update(pred, y)

        if self.on_debug_val is not None:
            self.on_debug_val(self, batch, batch_idx, logits)
    
        return loss

    def on_test_epoch_start(self):
        self.test_loss.reset()
        for metric in self.test_metrics.values():
            metric.reset()

    def on_test_epoch_end(self):
        self.log("test/loss", self.test_loss.compute())
        for name, metric in self.test_metrics.items():
            self.log(f"test/{name}", metric.compute())

    def test_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
            ):
        x, y = batch
        logits = self.forward(x)

        loss = self._calculate_loss(logits, y)
        self.test_loss.update(loss)
        self.log("test/loss", loss, on_step=True, on_epoch=False)

        pred = torch.argmax(logits, dim=1)
        for metric in self.test_metrics.values():
            metric.update(pred, y)

        if self.on_debug_test is not None:
            self.on_debug_test(self, batch, batch_idx, logits)
        
        return loss

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
    
    def export_onnx(self, filename: Union[str, Path]):
        assert hasattr(self, "example_input_array"),\
            "Exporting to ONNX requires 'example_input_array' attribute"
        
        ... # TODO
