#

import abc
from torch import nn
from typing import (
    Any,
    Iterable
)
from ..decoders.decoder import (
    Decoder,
    DecoderRegistry
)


class Encoder(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
    
    @property
    @abc.abstractmethod
    def backbone_params(self) -> Iterable[nn.Parameter]:
        ...

    @property
    @abc.abstractmethod
    def head_params(self) -> Iterable[nn.Parameter]:
        ...


class EncoderDecoder(nn.Module):
    decoder: Decoder

    def __init__(self):
        super().__init__()
    
    @classmethod
    def create_decoder(cls, decoder_name: str, *args, **kwargs) -> Decoder:
        return DecoderRegistry.create(decoder_name, *args, **kwargs)

    @property
    @abc.abstractmethod
    def backbone_params(self) -> Iterable[nn.Parameter]:
        ...
    
    @property
    def head_params(self) -> Iterable[nn.Parameter]:
        ...

    @abc.abstractmethod
    def forward_features(self, *args, **kwargs) -> Any:
        ...

    @abc.abstractmethod
    def forward_decoder(self, *args, **kwargs) -> Any:
        ...

    @abc.abstractmethod
    def forward_head(self, *args, **kwargs) -> Any:
        ...

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        ...


class EncoderMAE(nn.Module, abc.ABC):
    encoder: Encoder
    def __init__(self):
        super().__init__()
    
    @abc.abstractmethod
    def forward_encoder(self, *args, **kwargs) -> Any:
        ...
