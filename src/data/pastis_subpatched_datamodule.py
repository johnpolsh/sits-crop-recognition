#

import torch
import geopandas as gpd
from functools import partial
from lightning import LightningDataModule
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize
)
from typing import Any, Callable, Optional, TypedDict, Union
from .components.pastis import (
    _FOLDS,
    _SUBPATCHING_MODES,
    PASTISSubpatchedDatasetS2,
    load_data_mean_std
)
from .transforms import (
    DType,
    FromNumpy,
    Take,
    loose_bind_transform
)


class PastisParams(TypedDict):
    folds: _FOLDS
    subpatch_size: int
    metadata: Optional[Union[_FOLDS, gpd.GeoDataFrame, Callable[..., gpd.GeoDataFrame]]]
    transforms: list
    target_transforms: list
    subpatching_mode: _SUBPATCHING_MODES
    hparams: dict[str, Any]


class PastisSubpatchedDatamodule(LightningDataModule):
    _TRAIN_DEFAULTS: PastisParams = {
        "folds": [1, 2, 3],
        "subpatch_size": 3,
        "metadata": None,
        "subpatching_mode": "equidistant",
        "transforms": [
            loose_bind_transform(FromNumpy),
            loose_bind_transform(Normalize)
        ],
        "target_transforms": [
            loose_bind_transform(partial(Take, indices=0, dim=0)),
            loose_bind_transform(FromNumpy)
        ],
        "hparams": {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "shuffle": True,
            "persistent_workers": True,
        },
    }
    _VAL_DEFAULTS: PastisParams = {
        "folds": [4],
        "subpatch_size": 3,
        "metadata": None,
        "subpatching_mode": "equidistant",
        "transforms": [
            loose_bind_transform(FromNumpy),
            loose_bind_transform(Normalize)
        ],
        "target_transforms": [
            loose_bind_transform(partial(Take, indices=0, dim=0)),
            loose_bind_transform(FromNumpy),
            loose_bind_transform(partial(DType, dtype=torch.long))
        ],
        "hparams": {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "shuffle": False,
            "persistent_workers": True,
        },
    }

    def __init__(
            self,
            data_dir: Union[str, Path],
            train: Optional[PastisParams] = _TRAIN_DEFAULTS,
            val: Optional[PastisParams] = _VAL_DEFAULTS,
            test: Optional[PastisParams] = None
            ):
        super().__init__()
        self.data_dir = data_dir
        self.train = train
        self.val = val
        self.test = test
    
    def prepare_data(self):
        if self.train is not None:
            self.train_norm = load_data_mean_std(self.data_dir, self.train["folds"])

        if self.val is not None:
            self.val_norm = load_data_mean_std(self.data_dir, self.val["folds"])
        
        if self.test is not None:
            self.test_norm = load_data_mean_std(self.data_dir, self.test["folds"])

    def setup(self, stage: str):
        if stage == "fit":
            assert self.train is not None, "Train params must be provided"
            kwargs = {
                "mean": self.train_norm[0],
                "std": self.train_norm[1],
            }
            transform = Compose([
                transform(**kwargs) for transform in self.train["transforms"]
                ])
            target_transform = Compose([
                transform() for transform in self.train["target_transforms"]
                ])
            
            self.train_dataset = PASTISSubpatchedDatasetS2(
                data_dir=self.data_dir,
                subpatch_size=self.train["subpatch_size"],
                metadata=self.train["metadata"] if self.train["metadata"] is not None else self.train["folds"],
                subpatching_mode=self.train["subpatching_mode"],
                transform=transform,
                target_transform=target_transform
                )
        
        if stage in ["fit", "validate"]:
            assert self.val is not None, "Validation params must be provided"
            kwargs = {
                "mean": self.val_norm[0],
                "std": self.val_norm[1],
            }
            transform = Compose([
                transform(**kwargs) for transform in self.val["transforms"]
                ])
            target_transform = Compose([
                transform() for transform in self.val["target_transforms"]
                ])
            
            self.val_dataset = PASTISSubpatchedDatasetS2(
                data_dir=self.data_dir,
                subpatch_size=self.val["subpatch_size"],
                metadata=self.val["metadata"] if self.val["metadata"] is not None else self.val["folds"],
                subpatching_mode=self.val["subpatching_mode"],
                transform=transform,
                target_transform=target_transform
                )
        
        if stage == "test":
            assert self.test is not None, "Test params must be provided"
            kwargs = {
                "mean": self.test_norm[0],
                "std": self.test_norm[1],
            }
            transform = Compose([
                transform(**kwargs) for transform in self.test["transforms"]
                ])
            target_transform = Compose([
                transform() for transform in self.test["target_transforms"]
                ])
            
            self.test_dataset = PASTISSubpatchedDatasetS2(
                data_dir=self.data_dir,
                subpatch_size=self.test["subpatch_size"],
                metadata=self.test["metadata"] if self.test["metadata"] is not None else self.test["folds"],
                subpatching_mode=self.test["subpatching_mode"],
                transform=transform,
                target_transform=target_transform
                )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.train["hparams"] # type: ignore
            )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            **self.val["hparams"] # type: ignore
            )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            **self.test["hparams"] # type: ignore
            )
