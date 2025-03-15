#

import torch
import numpy as np
from typing import TypeVar


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
