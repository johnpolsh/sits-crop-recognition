#

import pandas as pd
import torch
import numpy as np
from typing import (
    Callable,
    Hashable
)
from .functional import (
    Transformable,
    hflip,
    mean,
    median,
    rotate90,
    vflip
)


class MultidataTransform:
    def __init__(
            self,
            keys: Hashable | list[Hashable] | None = None,
            ):
        self.keys = keys if isinstance(keys, list) else [keys] if keys is not None else None

    def transform(self, data: Transformable) -> Transformable:
        return data

    def __call__(self, data: dict) -> dict:
        if self.keys is None:
            return {k: self.transform(v) for k, v in data.items()}
        
        return {
            k: self.transform(v) if k in self.keys else v
            for k, v in data.items()
        }


class SelectiveMultidataTransform(MultidataTransform):
    def __init__(self, transform: Callable, keys: Hashable | list[Hashable]):
        super().__init__(keys)
        self.transform = transform


class ToTensor(MultidataTransform):
    def __init__(self, keys: Hashable | list[Hashable] | None = None):
        super().__init__(keys)

    def transform(self, data: Transformable) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

# TODO
class DType:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(self.dtype)
    

class TakeIndices(MultidataTransform):
    def __init__(
            self,
            indices: int | list[int],
            dim: int = 0,
            keys: Hashable | list[Hashable] | None = None
            ):
        super().__init__(keys)
        self.indices = tuple(indices) if isinstance(indices, list) else indices
        self.dim = dim

    def transform(self, data: Transformable) -> Transformable:
        if isinstance(data, torch.Tensor):
            return data.index_select(self.dim, torch.tensor(self.indices))
        else:
            return data.take(self.indices, axis=self.dim)


class Transpose(MultidataTransform):
    def __init__(
            self,
            dim0: int,
            dim1: int,
            keys: Hashable | list[Hashable] | None = None
            ):
        super().__init__(keys)
        self.dims = (dim0, dim1)

    def transform(self, data: Transformable) -> Transformable:
        if isinstance(data, torch.Tensor):
            return data.transpose(*self.dims)
        else:
            return data.swapaxes(*self.dims)


class Permute(MultidataTransform):
    def __init__(
            self,
            dims: tuple[int, ...],
            keys: Hashable | list[Hashable] | None = None
            ):
        super().__init__(keys)
        self.dims = dims

    def transform(self, data: Transformable) -> Transformable:
        if isinstance(data, torch.Tensor):
            return data.permute(self.dims)
        else:
            return data.transpose(*self.dims)
    

class Reshape(MultidataTransform):
    def __init__(
            self,
            shape: tuple[int, ...] | list[int],
            keys: Hashable | list[Hashable] | None = None
            ):
        super().__init__(keys)
        self.shape = tuple(shape)

    def transform(self, data: Transformable) -> Transformable:
        return data.reshape(self.shape)


class TemporalFeatureExtraction(MultidataTransform):
    def __init__(
            self,
            encoding_type: str = "doy",
            keys: Hashable | list[Hashable] | None = None
            ):
        super().__init__(keys)
        self.encoding_type = encoding_type

    def _as_doy(self, data: np.ndarray) -> np.ndarray:
        return pd.to_datetime(data).dayofyear.to_numpy()

    def transform(self, data: Transformable) -> Transformable:
        assert len(data.shape) == 1, f"Expected 1D data, got {data.shape}"
        assert np.issubdtype(type(data[0]), np.datetime64),\
            f"Expected data type to be np.datetime64, got {type(data[0])}"
        if isinstance(data, torch.Tensor):
            _data = data.numpy()
        else:
            _data = data

        if self.encoding_type == "doy":
            _data = self._as_doy(_data)
            _data = _data.reshape(-1, 1)
        
        _data = _data.astype(np.float32)
        
        if isinstance(data, torch.Tensor):
            return torch.from_numpy(_data)
        else:
            return _data
        

class RandomKRotation(MultidataTransform):
    def __init__(
            self,
            keys: Hashable | list[Hashable] | None = None
            ):
        super().__init__(keys)

    def __call__(self, data: dict) -> dict:
        k_rot = np.random.randint(4)
        if self.keys is None:
            return {k: rotate90(v, k_rot) for k, v in data.items()}
        
        return {
            k: rotate90(v, k_rot) if k in self.keys else v
            for k, v in data.items()
        }


class RandomHFlip(MultidataTransform):
    def __init__(
            self,
            p: float = 0.5,
            keys: Hashable | list[Hashable] | None = None
            ):
        super().__init__(keys)
        self.p = p

    def __call__(self, data: dict) -> dict:
        if np.random.rand() > self.p:
            if self.keys is None:
                return {k: hflip(v) for k, v in data.items()}
            return {
                k: hflip(v) if k in self.keys else v
                for k, v in data.items()
            }
        
        return data


class RandomVFlip(MultidataTransform):
    def __init__(
            self,
            p: float = 0.5,
            keys: Hashable | list[Hashable] | None = None
            ):
        super().__init__(keys)
        self.p = p

    def __call__(self, data: dict) -> dict:
        if np.random.rand() > self.p:
            if self.keys is None:
                return {k: vflip(v) for k, v in data.items()}
            return {
                k: vflip(v) if k in self.keys else v
                for k, v in data.items()
            }
        
        return data


class RandomTimestampAggregate:
    __choices__ = ["mean", "median", "pick"]
    def __init__(
        self,
        groups: int = 2,
        augments_probs: list[float] = [0.5, 0.5, 0.5]
        ):
        self.groups = groups
        self.augments_probs = augments_probs

    def __call__(self, data: Transformable) -> Transformable:
        assert data.ndim == 4, "Data must be 3D"
        assert data.shape[0] % self.groups == 0, "Timestamps dim must be divisible by groups"
        choice = np.random.choice(self.__choices__, p=self.augments_probs)
        _, C, H, W = data.shape
        if choice == "mean":
            data = data.reshape(self.groups, -1, C, H, W)
            return mean(data, axis=0)
        elif choice == "median":
            data = data.reshape(self.groups, -1, C, H, W)
            return median(data, axis=0)
        elif choice == "pick":
            indices = np.random.choice(
                data.shape[0],
                size=data.shape[0] // self.groups,
                replace=False
                )
            indices = np.sort(indices)
            return data[indices]
        
        return data
    