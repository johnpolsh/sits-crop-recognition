#

import torch
from torch.utils.data import Dataset
from typing import Union


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
        clip_min: float = 0.,
        clip_max: float = 1.,
        norm: bool = True
        ) -> torch.Tensor:
    bins, counts = torch.unique(targets, return_counts=True)
    weights = torch.zeros(num_classes, device=targets.device)
    weights[bins] = counts.max() / counts
    if norm:
        weights = weights / weights.max()
    weights = weights.clip(clip_min, clip_max)
    return weights


def compute_dataset_weight(
        dataset: Dataset,
        num_classes: int,
        clip_min: float = 0.,
        clip_max: float = 10.,
        device: Union[str, torch.device] = "cpu"
        ) -> torch.Tensor:
    targets = torch.cat([target.to(device) for _, target in dataset])
    return compute_class_weight(targets, num_classes, clip_min, clip_max)
