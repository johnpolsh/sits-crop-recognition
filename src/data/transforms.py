#

import torch
import numpy as np
from typing import Any, Callable, Union
from ..utils.scripting import loose_bind_kwargs


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
    def __init__(self, callback: Callable, steps: int):
        self.callback = callback
        self.steps = steps
        self._current_step = 0
        self._apply = False
    
    def step(self):
        self._current_step = (self._current_step + 1) % self.steps

    @property
    def apply(self) -> bool:
        if self._current_step == 0:
            self._apply = True
        return self._apply

    def __call__(self, data: Any) -> ...:
        if self.apply:
            data = self.callback(data)
        self.step()
        return data


class MultiStepRandomTransform(MultiStepTransform):
    def __init__(
            self,
            callback: Callable,
            steps: int,
            p: float = 0.5
            ):
        super().__init__(callback, steps=steps)
        self.p = p
    
    @property
    def apply(self) -> bool:
        if self._current_step == 0:
            self._apply = np.random.rand() < self.p
        return self._apply
