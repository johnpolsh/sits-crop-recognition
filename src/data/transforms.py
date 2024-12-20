#

import torch
import numpy as np
from typing import Any, Callable, Union
from ..utils.scripting import loose_bind_kwargs
from .functional import vflip, hflip, random_rotation


def loose_bind_transforms(
        transforms: list[Callable]
        ) -> list[Callable]:
    return [loose_bind_kwargs()(t) for t in transforms]


class FromNumpy:
    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data)


class DType:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(self.dtype)
    

class Take:
    def __init__(
            self,
            indices: Union[int, list[int]],
            dim: int = 0
            ):
        self.indices = tuple(indices) if isinstance(indices, list) else indices
        self.dim = dim

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return data.index_select(self.dim, torch.tensor(self.indices))
        else:
            return data.take(self.indices, axis=self.dim)


class Transpose:
    def __init__(self, dim0: int, dim1: int):
        self.dims = (dim0, dim1)

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return data.transpose(*self.dims)
        else:
            return data.transpose(self.dims)


class Reshape:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(shape)

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return data.reshape(self.shape)
    

class MultiStepTransform:
    def __init__(self, *callbacks: Callable, steps: int = -1):
        self.callbacks = callbacks
        self.steps = steps if steps > 0 else len(callbacks)
        self._current_step = 0
    
    def step(self):
        self._current_step = (self._current_step + 1) % self.steps

    def __call__(self, *data: Any) -> ...:
        data = self.callbacks[self._current_step](*data)
        self.step()
        return data


class CombinedRandomTransform(MultiStepTransform): # WANR: requires fix, does not work as intended
    def __init__(self, *callbacks: Callable, p: float = 0.5):
        super().__init__(*callbacks)
        self.p = p
        self._apply = False
    
    @property
    def apply(self) -> bool:
        if self._current_step == 0:
            self._apply = np.random.rand() < self.p
        return self._apply

    def __call__(self, *data: Any) -> ...:
        if self.apply:
            return super().__call__(*data)
        
        self.step()
        return data


class RandomHFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
            self,
            data: Union[np.ndarray, torch.Tensor]
            ) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p:
            return hflip(data)
        return data

class RandomVFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
            self,
            data: Union[np.ndarray, torch.Tensor]
            ) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p:
            return vflip(data)
        return data
