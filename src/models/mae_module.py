#

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib import colormaps
from torch import nn
from torchmetrics import (
    MeanSquaredError,
    PeakSignalNoiseRatio
)
from typing import Any, Literal, Optional, Union
from .base_module import BaseModule
from  .modules.functional import (
    depatchify_temporal
)
from ..utils.plotting import (
    img_tensor_to_numpy,
    make_grid_tensor,
    mask_tensor_to_rgb_tensor,
    normalize_img_tensor,
    pick_channels_tensor
)
from ..utils.scripting import extract_signature


def plot_prediction_patches(
        pl_module: BaseModule,
        batch,
        outputs,
        batch_idx: int = 0,
        prob_idx: Optional[Union[int, list[int]]] = None,
        idx: int = -1
        ):
    logits = outputs["logits"][idx].cpu().detach()
    targets = outputs["target"][idx].cpu().detach()

    logits = pl_module.net.unpatchify(logits.unsqueeze(dim=0))
    targets = pl_module.net.unpatchify(targets.unsqueeze(dim=0))

    channels = [2, 1, 0]
    x = logits[channels]
    x = normalize_img_tensor(x)
    x = img_tensor_to_numpy(x).astype(np.float32)
    y = targets[channels]
    y = normalize_img_tensor(y)
    y = img_tensor_to_numpy(y).astype(np.float32)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(x)
    ax[0].set_title("Predictions")
    ax[0].axis("off")
    ax[1].imshow(y)
    ax[1].set_title("Targets")
    ax[1].axis("off")
    
    fig.tight_layout()

    return fig


class MAEModule(BaseModule):
    def __init__(
            self,
            net: nn.Module,
            default_train_metrics: bool = True,
            default_val_metrics: bool = True,
            default_test_metrics: bool = True,
            **kwargs
            ):
        super().__init__(net=net, **kwargs)

        if default_train_metrics:
            self.train_metrics.add_module(
                "mse",
                MeanSquaredError()
                )
            self.train_metrics.add_module(
                "psnr",
                PeakSignalNoiseRatio()
                )
    
    def step(self, batch: dict):
        x = batch["data"]
        dates = batch.get("dates", None)
        loss, pred, y, idx_keep, idx_mask = self.net(x, dates)

        results = {
            "data": x,
            "logits": pred.detach(),
            "target": y,
            "loss": loss,
        }

        return results
