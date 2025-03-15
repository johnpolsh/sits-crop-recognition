#

import importlib
import inspect
from types import ModuleType
from typing import Any, Callable, Union
from .pylogger import RankedLogger


_logger = RankedLogger(__name__, rank_zero_only=True)


def dynamic_import(module: Union[str, ModuleType], name: str) -> Any:
    if isinstance(module, str):
        module = importlib.import_module(module)

    attr = getattr(module, name, None)
    if attr is None:
        raise ValueError(f"Attribute {name} not found")
    
    return attr


def extract_signature(cls: Callable) -> dict[str, inspect.Parameter]:
    if inspect.isclass(cls):
        signature = inspect.signature(cls.__init__)
    else:
        signature = inspect.signature(cls)
    signature = {k: v for k, v in signature.parameters.items() if k != "self"}
    return signature


def extract_default_args(cls: Callable) -> dict[str, Any]:
    signature = extract_signature(cls)
    defaults = {k: v.default for k, v in signature.items() if v.default is not inspect.Parameter.empty}
    return defaults


def loose_bind_kwargs(warn_unused: bool = False) -> Callable:
    def decorator(fn: Callable) -> Callable:
        sigature = extract_signature(fn)

        def wrapper(*args, **kwargs):
            matching = {k: v for k, v in kwargs.items() if k in sigature}
            if warn_unused:
                unused = set(kwargs.keys()) - set(matching.keys())
                if unused:
                    _logger.warning(f"Discarded kwarg: {unused}")
            return fn(*args, **matching)
        
        return wrapper
    return decorator

