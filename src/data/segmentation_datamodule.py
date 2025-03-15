#

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Optional, TypedDict, Union
from ..utils.pylogger import RankedLogger


_logger = RankedLogger(__name__, rank_zero_only=True)


_dataset = Union[Dataset, Callable[..., Dataset]]
class DatasetParams(TypedDict):
    dataset: _dataset
    kwargs: dict[str, Any]


class SegmentationDataModule(LightningDataModule):
    def __init__(
            self,
            train: Optional[Union[_dataset, DatasetParams]],
            val: Optional[Union[_dataset, DatasetParams]] = None,
            test: Optional[Union[_dataset, DatasetParams]] = None,
            **kwargs: Any
            ):
        super().__init__()
        self.kwargs = kwargs
        self.train = train
        self.val = val
        self.test = test

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
            **self.kwargs,
            **(self.train.get("kwargs", {}) if isinstance(self.train, dict) else {}) # type: ignore
            )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            **self.kwargs,
            **(self.val.get("kwargs", {}) if isinstance(self.val, dict) else {}) # type: ignore
            )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            **self.kwargs,
            **(self.test.get("kwargs", {}) if isinstance(self.test, dict) else {}) # type: ignore
            )
