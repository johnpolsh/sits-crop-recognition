#

import torch
from typing import Optional


def normalize(
        tensor: torch.Tensor,
        ignore_zeros: bool = True,
        eps: float = 1e-8
        ) -> torch.Tensor:
    if ignore_zeros:
        keep_mask = tensor != 0.
    else:
        keep_mask = torch.ones_like(tensor, dtype=torch.bool)
    mx = tensor[keep_mask].max()
    mn = tensor[keep_mask].min()
    tensor[keep_mask] = (tensor[keep_mask] - mn) / (mx - mn + eps)
    tensor = tensor.clamp(0., 1.)
    return tensor


def compute_class_weight(
        targets: torch.Tensor,
        num_classes: int,
        ignore_index: int = -100,
        clip_min: Optional[float] = 0.,
        clip_max: Optional[float] = None
        ) -> torch.Tensor:
    bins, counts = torch.unique(targets, return_counts=True)
    mask_idx = bins != ignore_index
    bins = bins[mask_idx]
    counts = counts[mask_idx]
    weights = torch.zeros(num_classes, device=targets.device)
    weights[bins] = counts.max() / counts
    weights = torch.clamp(weights, clip_min, clip_max)
    return weights
