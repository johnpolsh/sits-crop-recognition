#

import torch
import numpy as np
from typing import Any, TypeVar


Transformable = TypeVar("Transformable", np.ndarray, torch.Tensor)


def hflip(data: Transformable) -> Transformable:
    if isinstance(data, torch.Tensor):
        return data.flip(-1)
    else:
        return np.flip(data, axis=-1)


def normalize(data: Transformable, mean: Transformable, std: Transformable) -> Transformable:
    return (data - mean) / std


def rotate90(data: Transformable, k: int = 1) -> Transformable:
    if isinstance(data, torch.Tensor):
        return data.rot90(k=k, dims=(-2, -1))
    else:
        return np.rot90(data, k=k, axes=(-2, -1))


def vflip(data: Transformable) -> Transformable:
    if isinstance(data, torch.Tensor):
        return data.flip(-2)
    else:
        return np.flip(data, axis=-2)


def mean(data: Transformable, axis: Any = None, **kwargs) -> Transformable:
    if isinstance(data, torch.Tensor):
        return data.mean(dim=axis, **kwargs)
    else:
        return np.mean(data, axis=axis, **kwargs)


def median(data: Transformable, axis: Any = None, **kwargs) -> Transformable:
    if isinstance(data, torch.Tensor):
        return data.median(dim=axis, **kwargs)[0]
    else:
        return np.median(data, axis=axis, **kwargs)
    

def unique(data: Transformable, axis: Any = None, **kwargs) -> Transformable | tuple[Transformable, ...]:
    if isinstance(data, torch.Tensor):
        return torch.unique(data, dim=axis, **kwargs)
    else:
        return np.unique(data, axis=axis, **kwargs)
    