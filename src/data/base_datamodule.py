#

import os
from lightning import LightningDataModule
from torch.utils.data import (
    DataLoader,
    Dataset
)
from typing import (
    Any,
    Callable,
    TypedDict
)
from ..utils.pylogger import RankedLogger


_logger = RankedLogger(__name__, rank_zero_only=True)


_dataset = Dataset | Callable[..., Dataset]
class DatasetParams(TypedDict):
    dataset: _dataset
    kwargs: dict[str, Any]


class BaseDataModule(LightningDataModule):
    def __init__(
            self,
            train: _dataset | DatasetParams | None,
            val: _dataset | DatasetParams | None = None,
            test: _dataset | DatasetParams | None = None,
            **kwargs: Any
            ):
        super().__init__()
        self.kwargs = kwargs
        self.train = train
        self.val = val
        self.test = test

    def _prepocess_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        num_workers = kwargs.get("num_workers", 0)
        if isinstance(num_workers, float) and num_workers < 1.0:
            num_cores = os.cpu_count()
            assert num_cores is not None, "Cannot determine number of cores"
            num_workers = int(num_cores * num_workers)
            _logger.debug(f"num_workers is set to {num_workers}")
        
        kwargs["num_workers"] = int(num_workers)

        return kwargs

    def _get_kwargs_or_default(self, stage: str) -> Any:
        default_kwargs = self.kwargs

        if stage == "train":
            dataset_kwargs = self.train.get("kwargs", {}) if isinstance(self.train, dict) else {}
        elif stage == "val":
            dataset_kwargs = self.val.get("kwargs", {}) if isinstance(self.val, dict) else {}
        elif stage == "test":
            dataset_kwargs = self.test.get("kwargs", {}) if isinstance(self.test, dict) else {}
        else:
            dataset_kwargs = {}

        default_kwargs |= dataset_kwargs
        default_kwargs = self._prepocess_kwargs(default_kwargs)
        return default_kwargs

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage in ["fit", "train"]:
            assert self.train is not None, "Train params must be provided"

            if isinstance(self.train, Dataset):
                self.train_dataset = self.train
            elif isinstance(self.train, dict):
                assert "dataset" in self.train, "Train dataset must be provided"
                if isinstance(self.train["dataset"], Dataset):
                    self.train_dataset = self.train["dataset"]
                    _logger.warning("Dataset is already instantiated, ignoring default hparams")
                else:
                    self.train_dataset = self.train["dataset"]()
            else:
                self.train_dataset = self.train()
            
        if stage in ["fit", "validate"]:
            assert self.val is not None, "Validation params must be provided"
            
            if isinstance(self.val, Dataset):
                self.val_dataset = self.val
            elif isinstance(self.val, dict):
                assert "dataset" in self.val, "Validation dataset must be provided"
                if isinstance(self.val["dataset"], Dataset):
                    self.val_dataset = self.val["dataset"]
                    _logger.warning("Dataset is already instantiated, ignoring default hparams")
                else:
                    self.val_dataset = self.val["dataset"]()
            else:
                self.val_dataset = self.val()
        
        if stage in ["test"]:
            assert self.test is not None, "Test params must be provided"
            
            if isinstance(self.test, Dataset):
                self.test_dataset = self.test
            elif isinstance(self.test, dict):
                assert "dataset" in self.test, "Test dataset must be provided"
                if isinstance(self.test["dataset"], Dataset):
                    self.test_dataset = self.test["dataset"]
                    _logger.warning("Dataset is already instantiated, ignoring default hparams")
                else:
                    self.test_dataset = self.test["dataset"]()
            else:
                self.test_dataset = self.test()
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self._get_kwargs_or_default("train")
            )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            **self._get_kwargs_or_default("val")
            )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            **self._get_kwargs_or_default("test")
            )
