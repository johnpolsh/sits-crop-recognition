#

from torch import nn
from typing import Any, Callable, Iterable, Optional, Union


def named_apply(
        fn: Callable,
        module: nn.Module,
        name: str ='',
        depth_first: bool = True,
        include_root: bool = False
        ) -> nn.Module:
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
