#

import inspect
import numpy as np
import torch
from enum import Enum
from timm.layers.format import (
    Format,
    nchw_to,
    nhwc_to
)
from lightning.pytorch.trainer.states import RunningStage
from torch import nn
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Union
)


_int_or_tuple_2_t = int | tuple[int, int]
_int_or_tuple_3_t = int | tuple[int, int, int]


def _ntuple(n: int):
    def parse(x: Union[int, Iterable]) -> tuple:
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(np.repeat(x, n))
    return parse


to_1tuple = _ntuple(1) # type: Callable[..., tuple[Any]]
to_2tuple = _ntuple(2) # type: Callable[..., tuple[Any, Any]]
to_3tuple = _ntuple(3) # type: Callable[..., tuple[Any, Any, Any]]
to_4tuple = _ntuple(4) # type: Callable[..., tuple[Any, Any, Any, Any]]
to_ntuple = _ntuple


class InputFormat(str, Enum):
    NCHW = 'NCHW'
    NCDHW = 'NCDHW'


class Format3D(str, Enum):
    NCDHW = 'NCDHW'
    NDHWC = 'NDHWC'
    NCL = 'NCL'
    NLC = 'NLC'


def get_input_format(input: torch.Tensor) -> InputFormat:
    """
    Determine the input format of a tensor.

    Args:
        input (torch.Tensor): The input tensor.

    Returns:
        InputFormat: The format of the input tensor.
    """
    if len(input.shape) == 4:
        return InputFormat.NCHW
    elif len(input.shape) == 5:
        return InputFormat.NCDHW
    else:
        raise ValueError("Input tensor must be 4D or 5D.")


def ncdhw_to_nchw(input: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor from NCDHW format to NCHW format.

    Args:
        input (torch.Tensor): The input tensor in NCDHW format.

    Returns:
        torch.Tensor: The converted tensor in NCHW format.
    """
    B, _, _, H, W = input.shape
    return input.reshape(B, -1, H, W)


def ncdhw_to(x: torch.Tensor, fmt: Format3D) -> torch.Tensor:
    if fmt == Format3D.NDHWC:
        x = x.permute(0, 2, 3, 4, 1)
    elif fmt == Format3D.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format3D.NCL:
        x = x.flatten(2)
    return x


def ndhwc_to(x: torch.Tensor, fmt: Format3D) -> torch.Tensor:
    if fmt == Format3D.NCDHW:
        x = x.permute(0, 4, 1, 2, 3)
    elif fmt == Format3D.NLC:
        x = x.flatten(1, 2)
    elif fmt == Format3D.NCL:
        x = x.flatten(1, 2).transpose(1, 2)
    return x


def freeze_parameters(parameters: Iterable, freeze: bool) -> None:
    """
    Freezes or unfreezes parameters by setting requires_grad to `not freeze`.

    Args:
        parameters (Iterable): An iterable of parameters (e.g., model parameters) to be frozen or unfrozen.
        freeze (bool): If True, sets requires_grad to False, freezing the parameters. If False, sets requires_grad to True, unfreezing the parameters.
    """
    for param in parameters:
        param.requires_grad = not freeze


# NOTE: might move to utils
def extract_signature(cls: Any) -> dict[str, inspect.Parameter]:
    """
    Extracts the signature of a given function or class `__init__`, excluding the 'self' parameter.

    Args:
        cls (Any): The class or function to extract the signature from.

    Returns:
        dict[str, inspect.Parameter]: A dictionary where the keys are the parameter names and the values are the corresponding inspect.Parameter objects.
    """
    if inspect.isclass(cls):
        signature = inspect.signature(cls.__init__)
    else:
        signature = inspect.signature(cls)
    signature = {k: v for k, v in signature.parameters.items() if k != "self"}
    return signature


def extract_default_args(cls: Any) -> dict[str, Any]:
    """
    Extracts the default arguments from the given class's signature.

    Args:
        cls (Any): The class from which to extract default arguments.

    Returns:
        dict[str, Any]: A dictionary where the keys are the argument names and the values are the default values.
    """
    signature = extract_signature(cls)
    defaults = {k: v.default for k, v in signature.items() if v.default is not inspect.Parameter.empty}
    return defaults


def get_stage_name(stage: Optional[RunningStage]) -> str:
    """
    Get the name of the given stage.

    Args:
        stage (Optional[RunningStage]): The stage whose name is to be retrieved. If None, "unknown" will be returned.

    Returns:
        str: The name of the stage, or "unknown" if the stage is None.
    """
    return stage.value if stage else "unknown"


def named_apply(
        fn: Callable,
        module: nn.Module,
        name: str ='',
        depth_first: bool = True,
        include_root: bool = False
) -> nn.Module:
    """
    Applies a function to each torch module, optionally including the root module.

    Args:
        fn (Callable): The function to apply to each module. It should accept two arguments: 
                        the module itself and its name.
        module (nn.Module): The root module to start applying the function from.
        name (str, optional): The name of the root module. Defaults to an empty string.
        depth_first (bool, optional): If True, apply the function to child modules before the parent module.
                                        If False, apply the function to the parent module before child modules.
                                        Defaults to True.
        include_root (bool, optional): If True, include the root module in the application of the function.
                                        Defaults to False.

    Returns:
        nn.Module: The original module with the function applied to its submodules.
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True
            )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    N, C, H, W = images.shape
    assert H == W and H % patch_size == 0

    patch_h = patch_w = H // patch_size
    num_patches = patch_h * patch_w

    patches = images.reshape(shape=(N, C, patch_h, patch_size, patch_w, patch_size))
    patches = torch.einsum("nchpwq->nhwpqc", patches)
    patches = patches.reshape(shape=(N, num_patches, patch_size**2 * C))

    return patches


def unpatchify(
        patches: torch.Tensor,
        patch_size: int,
        channels: int = 3
        ) -> torch.Tensor:
    N, C = patches.shape[0], channels
    patch_h = patch_w = int(patches.shape[1] ** 0.5)
    assert patch_h * patch_w == patches.shape[1]

    images = patches.reshape(shape=(N, patch_h, patch_w, patch_size, patch_size, C))
    images = torch.einsum("nhwpqc->nchpwq", images)
    images = images.reshape(shape=(N, C, patch_h * patch_size, patch_h * patch_size))
    return images


def depatchify_temporal(
        x: torch.Tensor,
        patch_size: int,
        tubelet_size: int,
        img_size: int
        ) -> torch.Tensor:
    b, fhw, tpqc = x.shape

    h = w = img_size // patch_size
    f = fhw // (h * w)
    c = tpqc // (tubelet_size * patch_size**2)
    x = x.reshape(b, f, h, w, tubelet_size, patch_size, patch_size, c)
    x = torch.einsum("bfhwtpqc->bcfthpwq", x)
    x = x.reshape(b, c, f * tubelet_size, h * patch_size, w * patch_size)
    
    return x


def patchify_temporal(
        x: torch.Tensor,
        patch_size: int,
        tubelet_size: int
        ) -> torch.Tensor:
    b, c, t, h, w = x.shape

    f = t // tubelet_size

    w = h = h // patch_size
    x = x.reshape(b, c, f, tubelet_size, h, patch_size, w, patch_size)
    x = torch.einsum("bcfthpwq->bfhwtpqc", x)
    x = x.reshape(b, f * h * w, tubelet_size * patch_size**2 * c)

    return x
