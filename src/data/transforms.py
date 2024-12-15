#

import torch
import numpy as np
from typing import Callable, Union
from ..utils.scripting import loose_bind_kwargs


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
        self.indices = indices
        self.dim = dim

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            return data.index_select(self.dim, torch.tensor(self.indices))
        else:
            return data.take(self.indices, axis=self.dim)


class Reshape:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    def __call__(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return data.reshape(self.shape)
    

def loose_bind_transform(transform: Callable) -> Callable:
    dec_transform = loose_bind_kwargs()(transform)
    return dec_transform
