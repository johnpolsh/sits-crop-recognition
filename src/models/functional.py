#

import inspect
import numpy as np
import torch
from enum import Enum
from lightning.pytorch.trainer.states import RunningStage
from torch import nn
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Union
)


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


def freeze_parameters(parameters: Iterable, freeze: bool) -> None:
    """
    Freezes or unfreezes parameters by setting requires_grad to `not freeze`.

    Args:
        parameters (Iterable): An iterable of parameters (e.g., model parameters) to be frozen or unfrozen.
        freeze (bool): If True, sets requires_grad to False, freezing the parameters. If False, sets requires_grad to True, unfreezing the parameters.
    """
    for param in parameters:
        param.requires_grad = not freeze

# BUG: get_parameters is not working as expected
def get_parameters(
        module: nn.Module,
        include_only: list[str] = [],
        exclude: list[str] = []
        ) -> list[nn.Parameter]:
    """
    Retrieve parameters from a PyTorch module based on inclusion and exclusion criteria.

    Args:
        module (nn.Module): The PyTorch module from which to retrieve parameters.
        include_only (list[str], optional): List of parameter names to include. If empty, all parameters are considered.
        exclude (list[str], optional): List of parameter names to exclude.

    Returns:
        list[nn.Parameter]: List of parameters that match the inclusion and exclusion criteria.
    """
    if not include_only:
        params = []
        for full_name, param in module.named_parameters():
            name = full_name.split(".")[0]
            print(name)
            if name not in exclude:
                params.append(param)
    else:
        params = []
        for full_name, param in module.named_parameters():
            name = full_name.split(".")[0]
            print(name)
            if name in include_only and name not in exclude:
                params.append(param)
    return params

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
