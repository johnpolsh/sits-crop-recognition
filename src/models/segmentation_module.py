#

import math
import torch
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import colormaps
from torch import nn
from torchmetrics import (
    Accuracy,
    CohenKappa,
    JaccardIndex,
    Precision,
    Recall
)
from typing import Any, Literal, Optional, Union
from .base_module import BaseModule
from ..utils.plotting import (
    img_tensor_to_numpy,
    make_grid_tensor,
    mask_tensor_to_rgb_tensor,
    normalize_img_tensor,
    pick_channels_tensor
)
from ..utils.scripting import extract_signature


def plot_temporal_input(
        pl_module: BaseModule,
        batch,
        outputs,
        batch_idx: int = 0,
        num_channels: int = 3,
        idx: int = -1
        ):
    x = batch[0][idx].cpu().detach()
    x = x.reshape(-1, num_channels, x.shape[-2], x.shape[-1]).transpose(0, 1)
    F = x.shape[1]
    x = normalize_img_tensor(x)
    x = pick_channels_tensor(x, [2, 1, 0])
    x = make_grid_tensor(x, nrow=F, pad_value=1.)
    x = img_tensor_to_numpy(x)
    fig, ax = plt.subplots(1, 1, figsize=(8 * F, 8))
    ax.imshow(x)
    ax.axis("off")

    fig.tight_layout()

    return fig


def plot_temporal_prediction(
        pl_module: BaseModule,
        batch,
        outputs,
        batch_idx: int = 0,
        idx: int = -1
        ):
    y = batch["target"][idx].cpu().detach()
    colormap = colormaps.get_cmap("tab20")
    y_mask = mask_tensor_to_rgb_tensor(
        y,
        num_classes=pl_module.net.num_classes,
        colormap=colormap
        )
    y_mask = img_tensor_to_numpy(y_mask)

    logits = outputs["logits"][idx].cpu().detach()
    yhat = torch.argmax(logits, dim=0)
    yhat_mask = mask_tensor_to_rgb_tensor(
        yhat,
        num_classes=pl_module.net.num_classes,
        colormap=colormap
        )
    yhat_mask = img_tensor_to_numpy(yhat_mask)

    ignore = y == pl_module.ignore_index
    y_diff = y == yhat
    y_diff = y_diff.numpy().astype(float)
    y_diff[ignore] = 1.

    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].imshow(y_mask)
    ax[0].axis("off")
    ax[0].set_title("g_truth")

    ax[1].imshow(yhat_mask)
    ax[1].axis("off")
    ax[1].set_title("predict")

    ax[2].imshow(y_diff, cmap="gray")
    ax[2].axis("off")
    ax[2].set_title("diff")

    fig.tight_layout()

    return fig, f"predict-{batch_idx}.png"


def plot_temporal_prediction_prob(
        pl_module: BaseModule,
        batch,
        outputs,
        batch_idx: int = 0,
        prob_idx: Union[int, list[int]] | None = None,
        idx: int = -1
        ):
    logits = outputs["logits"][idx].cpu().detach()
    prob = torch.softmax(logits, dim=0)

    if prob_idx is None:
        prob_idx = list(range(prob.shape[0]))
    elif isinstance(prob_idx, int):
        prob_idx = [prob_idx]
    
    prob = prob[prob_idx]
    prob = prob.numpy()
    
    fig, ax = plt.subplots(1, prob.shape[0], figsize=(8 * prob.shape[0], 8))
    norm = mcolors.Normalize(vmin=0., vmax=1.)
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    for i in range(prob.shape[0]):
        ax[i].imshow(prob[i], vmin=0., vmax=1.)
        ax[i].axis("off")

    fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    fig.tight_layout()

    return fig, f"predict_prob-{batch_idx}.png"


def plot_prediction(
        pl_module: BaseModule,
        batch,
        outputs,
        batch_idx: int = 0,
        prob_idx: int = 1
        ):
    idx = -1
    x = batch["data"][idx].cpu().detach()
    x = pick_channels_tensor(x, [2, 1, 0])
    x = normalize_img_tensor(x)
    x = img_tensor_to_numpy(x)

    colormap = colormaps.get_cmap("tab20")
    y_b = batch["label"][idx].squeeze(0).cpu().detach()
    label_mask = y_b == 255

    y = y_b.clone()
    y[label_mask] = pl_module.net.num_classes
    y = mask_tensor_to_rgb_tensor(
        y,
        num_classes=pl_module.net.num_classes + 1,
        colormap=colormap
        )
    y = img_tensor_to_numpy(y)

    logits = outputs["logits"][idx].cpu().detach()

    yhat_b = torch.argmax(logits, dim=0)
    yhat = yhat_b.clone()
    yhat[label_mask] = pl_module.net.num_classes
    yhat = mask_tensor_to_rgb_tensor(
        yhat,
        num_classes=pl_module.net.num_classes + 1,
        colormap=colormap
        )
    yhat = img_tensor_to_numpy(yhat)

    y_diff = y_b == yhat_b
    y_diff = y_diff.numpy().astype(float)

    prob = torch.softmax(logits, dim=0)
    prob = prob[prob_idx].numpy()

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 4)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(x)
    ax.axis("off")
    ax.set_title("target")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(y)
    ax.axis("off")
    ax.set_title("g_truth")

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(yhat)
    ax.axis("off")
    ax.set_title("predict")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(y)
    ax.imshow(yhat, alpha=0.5)
    ax.axis("off")
    ax.set_title("overlay (.5)")

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(y_diff, cmap="gray")
    ax.axis("off")
    ax.set_title("diff")

    ax = fig.add_subplot(gs[1, 3])
    fig.colorbar(ax.imshow(prob, vmin=0., vmax=1.), ax=ax)
    ax.axis("off")
    ax.set_title(f"prob_idx ({prob_idx})")

    # up to 10 classes per column
    num_cols = math.ceil((pl_module.net.num_classes + 1) / 10)
    gs_sub = gridspec.GridSpecFromSubplotSpec(
        10,
        num_cols,
        subplot_spec=gs[0, 3]
        )
    classes = [*range(pl_module.net.num_classes), 255]
    for i in range(10):
        for j in range(num_cols):
            idx = i + j * 10
            if idx > pl_module.net.num_classes:
                break
            ax = fig.add_subplot(gs_sub[i, j])
            
            rect = patches.Rectangle(
                (0.0, 0.0),
                0.6,
                1.0,
                color=colormap(idx / pl_module.net.num_classes)
                )
            ax.add_patch(rect)
            ax.text(0.7, 0.5, f"{classes[idx]}", fontsize=10, verticalalignment="center")
            ax.axis("off")

    fig.tight_layout()

    return fig, f"plot_predict-{batch_idx}.png"


class SegmentationModule(BaseModule):
    def __init__(
            self,
            net: nn.Module,
            default_train_metrics: bool = True,
            default_val_metrics: bool = True,
            default_test_metrics: bool = True,
            **kwargs
            ):
        super().__init__(net=net, **kwargs)

        if isinstance(self.criterion, nn.Module):
            if hasattr(self.criterion, "mode"):
                metric_task = self.criterion.mode
            else:
                metric_task = "multiclass" if self.net.num_classes > 1 else "binary"

            if hasattr(self.criterion, "ignore_index"):
                ignore_index = self.criterion.ignore_index
            else:
                ignore_index = 255
        else:
            criterion_params = extract_signature(self.criterion)

            if "mode" in criterion_params:
                metric_task = criterion_params["mode"].default
            else:
                metric_task = "multiclass" if self.net.num_classes > 1 else "binary"
            
            if "ignore_index" in criterion_params:
                ignore_index = criterion_params["ignore_index"].default
            else:
                ignore_index = 255
        
        self.metric_task = metric_task
        self.ignore_index = ignore_index

        if default_train_metrics:
            self.train_metrics.add_module(
                "acc",
                Accuracy(
                    task=metric_task,
                    num_classes=net.num_classes,
                    ignore_index=ignore_index,
                    average="weighted"
                    )
                )
            self.train_metrics.add_module(
                "jacc",
                JaccardIndex(
                    task=metric_task,
                    num_classes=net.num_classes,
                    ignore_index=ignore_index,
                    average="weighted"
                    )
                )
    
        if default_val_metrics:
            self.val_metrics.add_module(
                "acc",
                Accuracy(
                    task=metric_task,
                    num_classes=net.num_classes,
                    ignore_index=ignore_index,
                    average="weighted"
                    )
                )
            self.val_metrics.add_module(
                "jacc",
                JaccardIndex(
                    task=metric_task,
                    num_classes=net.num_classes,
                    ignore_index=ignore_index,
                    average="weighted"
                    )
                )
            self.val_metrics.add_module(
                "kappa",
                CohenKappa(
                    task=metric_task,
                    num_classes=net.num_classes,
                    ignore_index=ignore_index
                    )
                )
        
        if default_test_metrics:
            self.test_metrics.add_module(
                "acc",
                Accuracy(
                    task=metric_task,
                    num_classes=net.num_classes,
                    ignore_index=ignore_index,
                    average="weighted"
                    )
                )
            self.test_metrics.add_module(
                "jacc",
                JaccardIndex(
                    task=metric_task,
                    num_classes=net.num_classes,
                    ignore_index=ignore_index,
                    average="weighted"
                    )
                )
            self.test_metrics.add_module(
                "kappa",
                CohenKappa(
                    task=metric_task,
                    num_classes=net.num_classes,
                    ignore_index=ignore_index
                    )
                )
            self.test_metrics.add_module(
                "precision",
                Precision(
                    task=metric_task,
                    num_classes=net.num_classes,
                    ignore_index=ignore_index,
                    average="weighted"
                    )
                )
            self.test_metrics.add_module(
                "recall",
                Recall(
                    task=metric_task,
                    num_classes=net.num_classes,
                    ignore_index=ignore_index,
                    average="weighted"
                    )
                )

    def step(self, batch: dict):
        x = batch["data"]
        y = batch["target"]
        dates = batch.get("dates", None)
        logits = self.net.forward(x, dates)

        results = {
            "data": x,
            "logits": logits.detach(),
            "target": y,
            "loss": self._compute_loss(logits, y),
        }

        return results
