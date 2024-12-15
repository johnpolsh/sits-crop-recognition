#

import torch
from torch.utils.data import Dataset
from typing import Union


def compute_class_weight(
        targets: torch.Tensor,
        num_classes: int,
        clip_max: float = 10.
        ) -> torch.Tensor:
    bins, counts = torch.unique(targets, return_counts=True)
    weights = torch.zeros(num_classes, device=targets.device)
    weights[bins] = counts.max() / counts
    weights = weights.clip(0., clip_max)
    return weights


def compute_dataset_weight(
        dataset: Dataset,
        num_classes: int,
        clip_max: float = 10.,
        device: Union[str, torch.device] = "cpu"
        ) -> torch.Tensor:
    targets = torch.cat([target.to(device) for _, target in dataset])
    return compute_class_weight(targets, num_classes, clip_max)
