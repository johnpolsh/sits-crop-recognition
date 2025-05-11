#

import torch
from torch import nn
from typing import (
    Callable,
    Literal
)
from ..functional import (
    Format3D,
    ncdhw_to,
    to_2tuple
)

def get_1d_sincos_embed_from_grid_torch(embed_dim: int, pos: torch.Tensor):
    assert embed_dim % 2 == 0,\
        f"Expected `embed_dim` to be even, got `{embed_dim}`"
    assert pos.dtype in [torch.float32, torch.float16, torch.bfloat16],\
        f"Expected `pos` to be of type `float32`, `float16` or `bfloat16`, got `{pos.dtype}`"

    omega = torch.arange(embed_dim // 2, dtype=pos.dtype).to(pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

    return emb


_temporal_encoding_type = Literal["doy"]
class TemporalEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            encoding_type: _temporal_encoding_type= "doy",
            trainable_scale: bool = False
            ):
        super().__init__()
        assert encoding_type in ["doy"], f"Unsupported encoding type `{encoding_type}`"
        assert embed_dim % 2 == 0, f"Expected `embed_dim` to be even, got `{embed_dim}`"
        self.embed_dim = embed_dim
        self.encoding_type = encoding_type
        
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))
    
    def get_doy_encoding(
            self,
            embed_dim: int,
            doy_coords: torch.Tensor
            ) -> torch.Tensor:
        return get_1d_sincos_embed_from_grid_torch(
            embed_dim,
            doy_coords
            )

    def forward(
            self,
            temporal_coords: torch.Tensor
            ) -> torch.Tensor:
        B, T, N = temporal_coords.shape

        if self.encoding_type == "doy":
            assert N == 1, f"Expected `{N}` to be equal to `1`"
            doy = temporal_coords[:, :, 0].flatten()
        
        doy = self.get_doy_encoding(
            self.embed_dim,
            temporal_coords[:, :, 0].flatten()
            ).reshape(B, T, -1)
        
        embed = self.scale * doy

        return embed


class PatchEmbed3D(nn.Module):
    output_fmt: Format3D
    
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 4,
            in_channels: int = 3,
            num_frames: int = 3,
            tubelet_size: int = 1,
            embed_dim: int = 96,
            norm_layer: Callable[..., nn.Module] | None = None,
            flatten: bool = True,
            output_fmt: str | None = None,
            bias: bool = True
            ):
        super().__init__()
        assert img_size % patch_size == 0,\
            "Image size must be divisible by patch size"
        assert num_frames % tubelet_size == 0,\
            "Number of frames must be divisible by tubelet size"

        self.patch_size = to_2tuple(patch_size)
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.img_size = to_2tuple(img_size)

        self.flatten = flatten if output_fmt is None else False
        self.output_fmt = Format3D(output_fmt) if output_fmt is not None else Format3D.NCDHW
        self.grid_size = (
            num_frames // tubelet_size,
            img_size // patch_size,
            img_size // patch_size
            )
        self.num_patches = (
            self.grid_size[0]
            * self.grid_size[1]
            * self.grid_size[2]
            )
        
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            bias=bias
            )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, H, W = x.shape
        assert H == self.img_size[0], f"Expected height `{self.img_size[0]}`, got `{H}`"
        assert W == self.img_size[1], f"Expected width `{self.img_size[1]}`, got `{W}`"

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        elif self.output_fmt == Format3D.NDHWC:
            x = ncdhw_to(x, self.output_fmt)
        x = self.norm(x)
        return x
