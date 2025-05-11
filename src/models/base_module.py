#

import torch
from lightning import LightningModule
from pathlib import Path
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torchmetrics import MeanMetric
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    TypedDict,
    Union
)
from ..utils.scripting import extract_default_args

def freeze_parameters(parameters: Iterable, freeze: bool) -> None:
    """
    Freezes or unfreezes parameters by setting requires_grad to `not freeze`.

    Args:
        parameters (Iterable): An iterable of parameters (e.g., model parameters) to be frozen or unfrozen.
        freeze (bool): If True, sets requires_grad to False, freezing the parameters. If False, sets requires_grad to True, unfreezing the parameters.
    """
    for param in parameters:
        param.requires_grad = not freeze


_criterion = Union[_Loss, Callable[..., _Loss]]
_optimizer = Callable[..., optim.Optimizer]
_scheduler = Callable[..., optim.lr_scheduler._LRScheduler]
_optimizer_strategy = Literal["default", "freeze", "step-freeze", "backbone-lr"]


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
            scheduler: Optional[_scheduler] = None,
            optimized_metric: str = "val/loss",
            optimizer_strategy: Optional[OptimizerStrategyParam] = None,
            ignore_hyparams: list[str] = ["net", "criterion"]
            ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=ignore_hyparams)

        # https://arxiv.org/abs/1904.06376
        torch.set_float32_matmul_precision("medium")
        
        if net.example_input_array is None:
            assert hasattr(net, "in_channels"),\
                "Expected net to have `in_channels` attribute"
            assert hasattr(net, "img_size"),\
                "Expected net to have `img_size` attribute"
            
            net.example_input_array = torch.randn(
                1,
                net.in_channels,
                net.img_size,
                net.img_size
                )
        self.example_input_array = net.example_input_array

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
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net
        self.optimized_metric = optimized_metric

        self.train_loss = MeanMetric()
        self.train_metrics = nn.ModuleDict()

        self.val_loss = MeanMetric()
        self.val_metrics = nn.ModuleDict()

        self.test_loss = MeanMetric()
        self.test_metrics = nn.ModuleDict()

    def _compute_loss(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
            ) -> torch.Tensor:
        if isinstance(self.criterion, nn.Module):
            return self.criterion(logits, targets)
        
        criterion = self.criterion()
        return criterion(logits, targets)
    
    def _configure_optimizer_strategy(self, default_lr: float) -> optim.Optimizer:
        assert self.optimizer_strategy is not None,\
            "Expected optimizer_strategy to be not None"
        
        backbone_lr = self.optimizer_strategy.get("backbone_lr")
        lr = backbone_lr if backbone_lr is not None else default_lr
            
        if self.optimizer_strategy["strategy"] in ["freeze"]:
            if "freeze_step" in self.optimizer_strategy and self.optimizer_strategy["freeze_step"] is not None:
                assert self.optimizer_strategy["freeze_step"] >= 0,\
                    f"Expected {self.optimizer_strategy['freeze_step']=} to be >= 0" # type: ignore
            else:
                freeze_parameters(self.net.backbone_params, True)

        if self.optimizer_strategy["strategy"] in ["step-freeze"]:
            assert "freeze_step" in self.optimizer_strategy,\
                f"Expected self.optimizer_strategy to have 'freeze_step' key"
            assert self.optimizer_strategy["freeze_step"] is not None,\
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
            if current_epoch >= self.optimizer_strategy.get("end", 0): # type: ignore
                freeze_parameters(self.net.backbone_params, True)
            elif current_epoch % freeze_epoch == 0: # type: ignore
                freeze_parameters(
                    self.net.backbone_params,
                    (current_epoch + freeze_epoch) % (freeze_epoch * 2) == 0 # type: ignore
                    )
        
        if self.optimizer_strategy["strategy"] in ["backbone-lr"]:
            if current_epoch >= self.optimizer_strategy.get("end", 0): # type: ignore
                freeze_parameters(self.net.backbone_params, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_epoch_start(self):
        self.train_loss.reset()
        for metric in self.train_metrics.values():
            metric.reset()
        
        if self.optimizer_strategy is not None:
            self._update_optimizer_strategy_state()

    def training_step(self, batch: Any, batch_idx: int):
        results = self.step(batch)
        loss = results["loss"]
        logits = results["logits"]
        target = results["target"]

        self.train_loss.update(loss)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            logger=False,
            prog_bar=True
            )

        for metric in self.train_metrics.values():
            metric.update(logits, target)

        return results

    def on_train_epoch_end(self):
        self.log(
            "train/loss",
            self.train_loss.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True
            )
        for name, metric in self.train_metrics.items():
            self.log(
                f"train/{name}",
                metric.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=True
                )

    def on_validation_epoch_start(self):
        self.val_loss.reset()
        for metric in self.val_metrics.values():
            metric.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        results = self.step(batch)
        loss = results["loss"]
        logits = results["logits"]
        target = results["target"]

        self.val_loss.update(loss)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=False,
            logger=False,
            prog_bar=True
            )

        for metric in self.val_metrics.values():
            metric.update(logits, target)

        return results

    def on_validation_epoch_end(self):
        self.log(
            "val/loss",
            self.val_loss.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True
            )
        for name, metric in self.val_metrics.items():
            self.log(
                f"val/{name}",
                metric.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=True
                )

    def on_test_epoch_start(self):
        self.test_loss.reset()
        for metric in self.test_metrics.values():
            metric.reset()

    def test_step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
            ):
        results = self.step(batch)
        loss = results["loss"]
        logits = results["logits"]
        target = results["target"]

        self.test_loss.update(loss)
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=False,
            logger=False,
            prog_bar=True
            )

        for metric in self.test_metrics.values():
            metric.update(logits, target)

        return results

    def on_test_epoch_end(self):
        self.log(
            "test/loss",
            self.test_loss.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True
            )
        for name, metric in self.test_metrics.items():
            self.log(
                f"test/{name}",
                metric.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=True
                )

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
                "strict": True,
                "monitor": self.optimized_metric,
                "name": "lr_scheduler",
            }
            scheduler.append(config)
        
        return [optimizer], scheduler
