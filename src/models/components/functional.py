#

import torch
from typing import Optional, Union


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.,
    eps: float = 1e-7,
    dim: Optional[Union[int, torch.Size, list[int], tuple[int, ...]]]=None
) -> torch.Tensor:
    assert output.shape == target.shape,\
        f"Expected {output.shape=} to be equal to {target.shape=}"

    if dim is not None:
        intersection = torch.sum(output * target, dim=dim)
        cardinality = torch.sum(output + target, dim=dim)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    dice_score = (2. * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score
