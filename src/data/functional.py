#

import torch
import numpy as np
from typing import Union


def vflip(
        data: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(data, torch.Tensor):
        return data.flip(-2)
    else:
        return np.flip(data, axis=-2)


def hflip(
        data: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(data, torch.Tensor):
        return data.flip(-1)
    else:
        return np.flip(data, axis=-1)


def rotate90(
        data: Union[np.ndarray, torch.Tensor],
        k: int = 1
        ) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(data, torch.Tensor):
        return data.rot90(k=k, dims=(-2, -1))
    else:
        return np.rot90(data, k=k, axes=(-2, -1))


def random_rotation(
        data: Union[np.ndarray, torch.Tensor],
        ) -> Union[np.ndarray, torch.Tensor]:
    k = np.random.randint(4)
    return rotate90(data, k=k)
