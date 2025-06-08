#

import torch
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import (
    Dataset,
    Subset
)
from typing import Any
from .base_datamodule import (
    BaseDataModule,
    DatasetParams,
    _dataset
)
from .functional import unique
from ..utils.pylogger import RankedLogger


_logger = RankedLogger(__name__, rank_zero_only=True)


def get_class_indices(dataset: Dataset, ignore_index: int = -1, min_items: int = 1) -> dict[int, list[int]]:
    class_indices = defaultdict(list)
    
    for idx in tqdm(range(len(dataset)), desc="Getting class indices"): # type: ignore
        if hasattr(dataset, "_get_target"):
            label = dataset._get_target(idx) # type: ignore
        else:
            sample = dataset[idx]
            if isinstance(sample, tuple):
                label = sample[1]
            else:
                label = sample["target"]
        
        classes, counts = unique(label, return_counts=True)
    
        for cls, count in zip(classes, counts):
            if cls == ignore_index:
                continue

            if min_items > 0 and count < min_items:
                continue

            class_indices[cls].append(idx)

    return class_indices


def sample_k_per_class(dataset_or_indices: Dataset | dict[int, list[int]], k: int, exclusive: bool = True) -> dict[int, list[int]]:
    if isinstance(dataset_or_indices, Dataset):
        class_indices = get_class_indices(dataset_or_indices)
    else:
        class_indices = dataset_or_indices
    
    used_indices = set()
    if exclusive:
        _logger.info("Sampling classes exclusively")
        exclusive_indices = {}
        for cls, indices in sorted(class_indices.items(), key=lambda x: -len(x[1]), reverse=True):
            filtered = list(set(indices) - used_indices)
            exclusive_indices[cls] = filtered
            used_indices.update(filtered)
        class_indices = exclusive_indices

    sampled = {}
    for cls, indices in class_indices.items():
        unique_indices = torch.unique(torch.tensor(indices))
        if len(unique_indices) < k:
            _logger.debug(f"Class {cls} has less than {k} items, sampling {len(unique_indices)} items")
            
        perm = torch.randperm(len(unique_indices))[:k]
        sampled_idxs = unique_indices[perm].tolist()
        sampled[cls] = sampled_idxs
    
    return sampled

class FewShotDataModule(BaseDataModule):
    def __init__(
            self,
            train: _dataset | DatasetParams | None,
            val: _dataset | DatasetParams | None = None,
            disable_val: bool = False,
            k_shot: int = 1,
            min_items: int = 1,
            exclusive: bool = True,
            **kwargs: Any
            ):
        super().__init__(train, val, disable_val, test=None, **kwargs)
        self.k_shot = k_shot
        self.min_items = min_items
        self.exclusive = exclusive

    def _get_subset_indices(self, dataset: Dataset) -> list[int]:
        class_indices = get_class_indices(dataset, min_items=self.min_items)
        sampled = sample_k_per_class(class_indices, self.k_shot, self.exclusive)
        unique_indexes = set()
        for indices in sampled.values():
            unique_indexes.update(indices)
        return list(unique_indexes)

    def setup(self, stage: str):
        super().setup(stage)
        if stage in ["fit", "train"] and self.k_shot >= 0:
            self.train_dataset = Subset(self.train_dataset, indices=self._get_subset_indices(self.train_dataset))
