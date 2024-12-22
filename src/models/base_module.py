#

import torch
from lightning import LightningModule
from pathlib import Path
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torchmetrics import MeanMetric
from typing import Any, Callable, Iterable, Literal, Optional, TypedDict, Union
from ..data.preprocessing import compute_class_weight
from ..utils.scripting import extract_default_args


_criterion = Union[_Loss, Callable[..., _Loss]]
_optimizer = Callable[..., optim.Optimizer]
_scheduler = Optional[Callable[..., optim.lr_scheduler._LRScheduler]]
_on_debug_hook = Callable[[Any, Any, int, torch.Tensor], None]
_class_weight_strategy = Literal["none", "per-batch"]
_optimizer_strategy = Literal["default", "freeze", "step-freeze", "backbone-lr"]


def freeze_parameters(parameters: Iterable, freeze: bool) -> None:
    for param in parameters:
        param.requires_grad = not freeze


def get_parameters(
        module: nn.Module,
        include_only: list[str] = [],
        exclude: list[str] = []
        ) -> list[nn.Parameter]:
    if not include_only:
        params = [
            param for name, param in module.named_parameters()
            if name not in exclude
        ]
    else:
        params = [
            param for name, param in module.named_parameters()
            if name in include_only and name not in exclude
        ]
    return params


class MetricParam(TypedDict):
    alias: Optional[str]
    metric: Callable[..., nn.Module]


class OptimizerStrategyParam(TypedDict):
    strategy: _optimizer_strategy
    freeze_step: Optional[int]
    backbone_lr: Optional[float]
    end: Optional[int]
    backbone_hparams: Optional[dict[str, Any]]
    head_hparams: Optional[dict[str, Any]]


class BaseModule(LightningModule):
    def __init__(
            self,
            net: nn.Module,
            criterion: _criterion = nn.CrossEntropyLoss(),
            optimizer: _optimizer = optim.Adam,
            scheduler: _scheduler = None,
            train_metrics: list[MetricParam] = [],
            val_metrics: list[MetricParam] = [],
            test_metrics: list[MetricParam] = [],
            on_debug_train: Optional[_on_debug_hook] = None,
            on_debug_val: Optional[_on_debug_hook] = None,
            on_debug_test: Optional[_on_debug_hook] = None,
            class_weight_strategy: _class_weight_strategy = "none",
            optimizer_strategy: Optional[OptimizerStrategyParam] = None,
            ignore_hyparams: list[str] = ["net", "criterion"]
            ):
        assert class_weight_strategy in ["none", "per-batch"],\
            f"Got {class_weight_strategy=}, expected one of ['none', 'per-batch']"

        super().__init__()
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.on_debug_train = on_debug_train
        self.on_debug_val = on_debug_val
        self.on_debug_test = on_debug_test

        torch.set_float32_matmul_precision("medium")

        assert hasattr(net, "num_classes"),\
            "Expected net to have num_classes attribute"
        self.class_weight_strategy = class_weight_strategy

        if optimizer_strategy is not None:
            assert optimizer_strategy["strategy"] in ["default", "freeze", "step-freeze", "backbone-lr"],\
                f"Got {optimizer_strategy['strategy']=}, expected one of ['default', 'freeze', 'step-freeze', 'backbone-lr']"
            if optimizer_strategy["strategy"] in ["step-freeze"]:
                assert optimizer_strategy.get("freeze_step") is not None,\
                    f"Expected freeze_step to be not None"
            if optimizer_strategy["strategy"] in ["backbone-lr"]:
                assert optimizer_strategy.get("backbone_lr") is not None,\
                    f"Expected backbone_lr to be not None"
            if optimizer_strategy["strategy"] in ["freeze", "step-freeze", "backbone-lr"]:
                assert hasattr(net, "backbone_params"),\
                    "Expected net to have backbone_params attribute"
                assert hasattr(net, "head_params"),\
                    "Expected net to have head_params attribute"
        self.optimizer_strategy = optimizer_strategy

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
            weight = compute_class_weight(
                y,
                self.net.num_classes,
                clip_max=self.net.num_classes * 1.5
                )
            criterion = self.criterion(weight=weight)
        else:
            criterion = self.criterion()
            
        return criterion(logits, y)
    
    def _configure_optimizer_strategy(self, default_lr: float) -> optim.Optimizer:
        assert self.optimizer_strategy is not None,\
            "Expected optimizer_strategy to be not None"
        
        backbone_lr = self.optimizer_strategy.get("backbone_lr")
        lr = backbone_lr if backbone_lr is not None else default_lr
            
        if self.optimizer_strategy["strategy"] in ["freeze"]:
            if self.optimizer_strategy.get("freeze_step") is not None:
                assert self.optimizer_strategy["freeze_step"] >= 0,\
                    f"Expected {self.optimizer_strategy['freeze_step']=} to be >= 0"
            else:
                freeze_parameters(self.net.backbone_params, True)

        if self.optimizer_strategy["strategy"] in ["step-freeze"]:
            assert self.optimizer_strategy.get("freeze_step") is not None,\
                f"Expected {self.optimizer_strategy['freeze_step']=} to be not None"
            assert self.optimizer_strategy["freeze_step"] > 0,\
                f"Expected {self.optimizer_strategy['freeze_step']=} to be > 0"
        
        # NOTE: convention, could be same as step-freeze with step=0
        if self.optimizer_strategy["strategy"] in ["backbone-lr"]:
            assert self.optimizer_strategy["backbone_lr"] is not None,\
                f"Expected {self.optimizer_strategy['backbone_lr']=} to be not None"
        
        optimizer = self.optimizer([
            {
                "params": self.net.backbone_params,
                "lr": lr,
                **self.optimizer_strategy.get("backbone_hparams", {}), # type: ignore
            },
            {
                "params": self.net.head_params,
                **self.optimizer_strategy.get("head_hparams", {}) # type: ignore
            }
        ], lr=default_lr)
        
        return optimizer

    def _update_optimizer_strategy_state(self):
        assert self.optimizer_strategy is not None,\
            "Expected optimizer_strategy to be not None"
        
        current_epoch = self.current_epoch + 1
        freeze_epoch = self.optimizer_strategy.get("freeze_step", 0)
        if self.optimizer_strategy["strategy"] in ["freeze"]:
            if current_epoch >= freeze_epoch: # type: ignore
                freeze_parameters(self.net.backbone_params, True)

        if self.optimizer_strategy["strategy"] in ["step-freeze"]:
            if current_epoch >= self.optimizer_strategy.get("end", 0):
                freeze_parameters(self.net.backbone_params, True)
            elif current_epoch % freeze_epoch == 0: # type: ignore
                freeze_parameters(
                    self.net.backbone_params,
                    (current_epoch + freeze_epoch) % (freeze_epoch * 2) == 0 # type: ignore
                    )

    def on_train_epoch_start(self):
        self.train_loss.reset()
        for metric in self.train_metrics.values():
            metric.reset()
        
        if self.optimizer_strategy is not None:
            self._update_optimizer_strategy_state()

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

        if self.on_debug_train is not None:
            self.on_debug_train(self, batch, batch_idx, logits)

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
        self.log("val/loss", loss, on_step=True, on_epoch=False)

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
        if self.optimizer_strategy is not None:
            optimizer = self._configure_optimizer_strategy(
                extract_default_args(self.optimizer)["lr"]
                )
        else:
            optimizer = self.optimizer(self.net.parameters())

        scheduler = []
        if self.scheduler is not None:
            config = {
                "scheduler": self.scheduler(optimizer),
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
