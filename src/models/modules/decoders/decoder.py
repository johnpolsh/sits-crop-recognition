#

import abc
import torch
from typing import Iterable
from torch import nn


class Decoder(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @property
    @abc.abstractmethod
    def decoder_params(self) -> Iterable[nn.Parameter]:
        """
        Returns the parameters of the decoder.
        """
        ...
    
    @abc.abstractmethod
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the decoder.
        """
        ...


class DecoderRegistry:
    _registry = {}

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Decoder:
        """
        Create a decoder instance by name.
        """
        if name not in cls._registry:
            raise ValueError(f"Decoder '{name}' not found in registry.")
        decoder_cls = cls._registry[name]
        return decoder_cls(*args, **kwargs)

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a decoder class.
        """
        def decorator(decoder_cls):
            cls._registry[name] = decoder_cls
            return decoder_cls
        return decorator
    
    @classmethod
    def get_decoder(cls, name: str) -> type[Decoder]:
        """
        Get a decoder class by name.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Decoder '{name}' not found in registry. Available decoders: {cls.get_all_decoders()}"
                )
        return cls._registry[name]
    
    @classmethod
    def get_all_decoders(cls):
        """
        Get all registered decoders.
        """
        return list(cls._registry.keys())
