#

import torch
import numpy as np
from typing import Any, Callable, Optional, Union
from .functional import (
    Transformable,
    hflip,
    mean,
    median,
    rotate90,
    vflip
    )
from ..utils.scripting import loose_bind_kwargs


class ToTensor:
    @loose_bind_kwargs()
    def __init__(self):
        pass

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data)

# TODO
class DType:
    @loose_bind_kwargs()
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(self.dtype)
    

class TakeIndices:
    @loose_bind_kwargs()
    def __init__(
            self,
            indices: Union[int, list[int]],
            dim: int = 0
            ):
        self.indices = tuple(indices) if isinstance(indices, list) else indices
        self.dim = dim

    def __call__(self, data: Transformable) -> Transformable:
        if isinstance(data, torch.Tensor):
            return data.index_select(self.dim, torch.tensor(self.indices))
        else:
            return data.take(self.indices, axis=self.dim)


class Transpose:
    @loose_bind_kwargs()
    def __init__(self, dim0: int, dim1: int):
        self.dims = (dim0, dim1)

    def __call__(self, data: Transformable) -> Transformable:
        if isinstance(data, torch.Tensor):
            return data.transpose(*self.dims)
        else:
            return data.transpose(self.dims)


class Reshape:
    @loose_bind_kwargs()
    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(shape)

    def __call__(self, data: Transformable) -> Transformable:
        return data.reshape(self.shape)


class MultidataTransform:
    @loose_bind_kwargs()
    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform
        pass

    def __call__(self, data: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(self.transform(d) for d in data) if self.transform is not None else data


class SelectiveMultidataTransform(MultidataTransform):
    @loose_bind_kwargs()
    def __init__(self, transform: Callable, indices: Union[int, list[int]]):
        super().__init__(transform)
        self.indices = [indices] if isinstance(indices, int) else indices
    
    def __call__(self, data: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            self.transform(data[i]) if i in self.indices else data[i] for i in range(len(data)) # type: ignore
            )


class RandomKRotation(MultidataTransform):
    @loose_bind_kwargs()
    def __init__(self):
        super().__init__()

    def __call__(self, data: tuple[Transformable, ...]) -> tuple[Transformable, ...]:
        k = np.random.randint(4)
        return tuple(rotate90(d, k) for d in data)


class RandomHFlip(MultidataTransform):
    @loose_bind_kwargs()
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def __call__(self, data: tuple[Transformable, ...]) -> tuple[Transformable, ...]:
        if np.random.rand() > self.p:
            return tuple(hflip(d) for d in data)
        return data


class RandomVFlip(MultidataTransform):
    @loose_bind_kwargs()
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def __call__(self, data: tuple[Transformable, ...]) -> tuple[Transformable, ...]:
        if np.random.rand() > self.p:
            return tuple(vflip(d) for d in data)
        return data


class RandomTimestampAggregate:
    __choices__ = ["mean", "median", "pick"]
    @loose_bind_kwargs()
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
    