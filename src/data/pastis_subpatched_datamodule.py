#

import geopandas as gpd
from functools import partial
from lightning import LightningDataModule
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from typing import Any, Callable, Optional, TypedDict, Union
from .components.pastis import (
    _FOLDS,
    _SUBPATCHING_MODE,
    PASTISSubpatchedDatasetS2,
    load_data_mean_std
)


class PastisParams(TypedDict):
    folds: _FOLDS
    subpatch_size: int
    metadata: Optional[Union[_FOLDS, gpd.GeoDataFrame, Callable[..., gpd.GeoDataFrame]]]
    transforms: Union[list, tuple]
    target_transforms: Union[list, tuple]
    shared_transforms: Union[list, tuple]
    subpatching_mode: _SUBPATCHING_MODE
    hparams: dict[str, Any]


class PastisSubpatchedDatamodule(LightningDataModule):
    def __init__(
            self,
            data_dir: Union[str, Path],
            train: Optional[PastisParams],
            val: Optional[PastisParams],
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
                t(**kwargs) if isinstance(t, partial) else t
                for t in self.train.get("transforms", [])
                ] + [
                t(**kwargs) if isinstance(t, partial) else t
                for t in self.train.get("shared_transforms", [])
                ])
            target_transform = Compose([
                t() if isinstance(t, partial) else t
                for t in self.train.get("target_transforms", [])
                ] + [
                t() if isinstance(t, partial) else t
                for t in self.train.get("shared_transforms", [])
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
                t(**kwargs) if isinstance(t, partial) else t
                for t in self.val.get("transforms", [])
                ] + [
                t(**kwargs) if isinstance(t, partial) else t
                for t in self.val.get("shared_transforms", [])
                ])
            target_transform = Compose([
                t() if isinstance(t, partial) else t
                for t in self.val.get("target_transforms", [])
                ] + [
                t() if isinstance(t, partial) else t
                for t in self.val.get("shared_transforms", [])
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
                t(**kwargs) if isinstance(t, partial) else t
                for t in self.test.get("transforms", [])
                ] + [
                t(**kwargs) if isinstance(t, partial) else t
                for t in self.test.get("shared_transforms", [])
                ])
            target_transform = Compose([
                t() if isinstance(t, partial) else t
                for t in self.test.get("target_transforms", [])
                ] + [
                t() if isinstance(t, partial) else t
                for t in self.test.get("shared_transforms", [])
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
