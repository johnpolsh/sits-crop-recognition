#

import itertools
import numpy as np
import torch
from enum import Enum
from timm.layers.classifier import ClassifierHead
from timm.layers.create_act import get_act_layer
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from timm.layers.typing import LayerType
from timm.models import register_notrace_function
from timm.models.vision_transformer import get_init_weights_vit
from timm.models.swin_transformer_v2 import SwinTransformerV2Block
from torch import nn
from torch.nn import functional as F
from typing import (
    Callable,
    Literal,
    Optional
)
from ..functional import (
    InputFormat,
    named_apply,
    to_2tuple,
    to_3tuple
)
from .decoders.unet import UnetDecoder


_int_or_tuple_3_t = int | tuple[int, int, int]


class Format3D(str, Enum):
    NCDHW = 'NCDHW'
    NDHWC = 'NDHWC'
    NCL = 'NCL'
    NLC = 'NLC'


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


def window_partition_3d(x: torch.Tensor, window_size: tuple[int, int, int]) -> torch.Tensor:
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0], window_size[0],
        H // window_size[1], window_size[1],
        W // window_size[2], window_size[2],
        C
        )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


@register_notrace_function
def window_reverse_3d(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    img_size: tuple[int, int, int]
    ) -> torch.Tensor:
    D, H, W = img_size
    C = windows.shape[-1]
    x = windows.view(
        -1,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        C
        )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(-1, D, H, W, C)
    return x


class PatchEmbed3D(nn.Module):
    output_fmt: Format3D
    #dynamic_img_pad: torch.jit.Final[bool]
    
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 4,
            in_channels: int = 3,
            num_frames: int = 3,
            tubelet_size: int = 1,
            embed_dim: int = 768,
            norm_layer: Callable[..., nn.Module] | None = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False, # TODO
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
        self.num_patches = np.prod(self.grid_size)
        
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            bias=bias
            )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

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


class PatchMerging(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_scale: int = 2,
            norm_layer: Callable = nn.LayerNorm
            ):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.out_dim = dim_scale * dim
        self.norm = norm_layer(dim * dim_scale**2)
        self.reduction = nn.Linear(dim * dim_scale**2, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        assert H % self.dim_scale == 0,\
            f"Expected height `{H}` to be divisible by `{self.dim_scale}`"
        assert W % self.dim_scale == 0,\
            f"Expected width `{W}` to be divisible by `{self.dim_scale}`"
        x = x.reshape(
            B,
            H // self.dim_scale,
            self.dim_scale,
            W // self.dim_scale,
            self.dim_scale,
            C
            )
        x = x.permute(0, 1, 3, 2, 4, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x
    

class SwinTransformerV2Stage(nn.Module):
    def __init__(
            self,
            dim: int,
            input_resolution: int | tuple[int, int],
            depth: int,
            num_heads: int,
            window_size: int,
            downscale: bool = True,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float | list[float] = 0.,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm
            ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            SwinTransformerV2Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer # type: ignore
                ) for i in range(depth)
            ])
        self.merging = PatchMerging(
            dim,
            dim_scale=2,
            norm_layer=norm_layer
            ) if downscale else nn.Identity()

    def forward(
            self,
            x: torch.Tensor
            ) -> torch.Tensor:
        x = self.blocks(x)
        x = self.merging(x)
        return x


def _get_1d_sincos_embed_from_grid_torch(embed_dim: int, pos: torch.Tensor):
    assert embed_dim % 2 == 0
    assert pos.dtype in [torch.float32, torch.float16, torch.bfloat16]

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
    
    def _get_doy_encoding(
            self,
            embed_dim: int,
            doy_coords: torch.Tensor
            ) -> torch.Tensor:
        return _get_1d_sincos_embed_from_grid_torch(
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
        
        doy = self._get_doy_encoding(
            self.embed_dim,
            temporal_coords[:, :, 0].flatten()
            ).reshape(B, T, -1)
        
        embed = self.scale * doy

        return embed


class WindowAttention3D(nn.Module):
    def __init__(
            self,
            dim: int = 224,
            window_size: _int_or_tuple_3_t = (1, 7, 7),
            num_heads: int = 8,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            pretrained_window_size: _int_or_tuple_3_t = (0, 0, 0)
            ):
        super().__init__()
        self.dim = dim
        self.window_size = to_3tuple(window_size)
        self.pretrained_window_size = to_3tuple(pretrained_window_size)
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # get relative_coords_table
        relative_coords_d = torch.arange(
            -(self.window_size[0] - 1),
            self.window_size[0],
            dtype=torch.float32
            )
        relative_coords_h = torch.arange(
            -(self.window_size[1] - 1),
            self.window_size[1],
            dtype=torch.float32
            )
        relative_coords_w = torch.arange(
            -(self.window_size[2] - 1),
            self.window_size[2],
            dtype=torch.float32
            )
        relative_coords = torch.stack(
            torch.meshgrid([relative_coords_d, relative_coords_h, relative_coords_w])
            )
        relative_coords_table = relative_coords.permute(1, 2, 3, 0).contiguous().unsqueeze(0)
        eps = 1e-6
        if self.pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (self.pretrained_window_size[0] - (1 - eps))
            relative_coords_table[:, :, :, 1] /= (self.pretrained_window_size[1] - (1 - eps))
            relative_coords_table[:, :, :, 2] /= (self.pretrained_window_size[2] - (1 - eps))
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - (1 - eps))
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - (1 - eps))
            relative_coords_table[:, :, :, 2] /= (self.window_size[2] - (1 - eps))
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer('k_bias', torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wd*Wh*Ww, Wd*Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) # type: ignore[assignment]
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=np.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1
            )  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(
            self,
            dim: int,
            input_resolution: _int_or_tuple_3_t,
            num_heads: int,
            window_size: _int_or_tuple_3_t = (1, 7, 7),
            shift_size: _int_or_tuple_3_t = 0,
            always_partition: bool = False,
            dynamic_mask: bool = False,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: LayerType = "gelu",
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            pretrained_window_size: _int_or_tuple_3_t = 0
            ):
        super().__init__()
        self.dim = dim
        self.input_resolution = to_3tuple(input_resolution)
        self.num_heads = num_heads
        self.target_shift_size = to_3tuple(shift_size)
        self.always_partition = always_partition
        self.dynamic_mask = dynamic_mask
        self.window_size, self.shift_size = self._calc_window_shift(window_size, shift_size)
        self.window_area = self.window_size[0] * self.window_size[1] * self.window_size[2]
        self.mlp_ratio = mlp_ratio

        self.attn = WindowAttention3D(
            dim,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            pretrained_window_size=pretrained_window_size,
            )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=get_act_layer(act_layer), # type: ignore[arg-type]
            drop=proj_drop,
            )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.register_buffer(
            "attn_mask",
            None if self.dynamic_mask else self.get_attn_mask(),
            persistent=False,
            )
    
    def get_attn_mask(self, x: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            if x is None:
                img_mask = torch.zeros((1, *self.input_resolution, 1))
            else:
                img_mask = torch.zeros(
                    (1, x.shape[1], x.shape[2], x.shape[3], 1),
                    dtype=x.dtype, device=x.device
                    )
            cnt = 0
            for d in (
                    (0, -self.window_size[0]),
                    (-self.window_size[0], -self.shift_size[0]),
                    (-self.shift_size[0], None),
                    ):
                for h in (
                        (0, -self.window_size[1]),
                        (-self.window_size[1], -self.shift_size[1]),
                        (-self.shift_size[1], None),
                        ):
                    for w in (
                            (0, -self.window_size[2]),
                            (-self.window_size[2], -self.shift_size[2]),
                            (-self.shift_size[2], None),
                            ):
                        img_mask[:, d[0]:d[1], h[0]:h[1], w[0]:w[1], :] = cnt
                        cnt += 1
            mask_windows = window_partition_3d(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask
    
    def _calc_window_shift(
            self,
            target_window_size: _int_or_tuple_3_t,
            target_shift_size: Optional[_int_or_tuple_3_t] = None
            ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        target_window_size = to_3tuple(target_window_size)
        if target_shift_size is None:
            # if passed value is None, recalculate from default window_size // 2 if it was active
            target_shift_size = self.target_shift_size
            if any(target_shift_size):
                # if there was previously a non-zero shift, recalculate based on current window_size
                target_shift_size = (
                    target_window_size[0],
                    target_window_size[1] // 2,
                    target_window_size[2] // 2
                    )
        else:
            target_shift_size = to_3tuple(target_shift_size)

        if self.always_partition:
            return target_window_size, to_3tuple(target_shift_size)

        target_window_size = to_3tuple(target_window_size)
        target_shift_size = to_3tuple(target_shift_size)
        window_size = [
            r if r <= w else w for r, w in zip(
                self.input_resolution,
                target_window_size
                )
        ]
        shift_size = [
            0 if r <= w else s for r, w, s in zip(
                self.input_resolution,
                window_size,
                target_shift_size
                )
        ]
        return to_3tuple(window_size), to_3tuple(shift_size)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        _, D, H, W, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3)
                )
        else:
            shifted_x = x

        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        shifted_x = nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dp, Hp, Wp, _ = shifted_x.shape

        # partition windows
        x_windows = window_partition_3d(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_area, C)

        # W-MSA/SW-MSA
        if getattr(self, 'dynamic_mask', False):
            attn_mask = self.get_attn_mask(shifted_x)
        else:
            attn_mask = self.attn_mask
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = window_reverse_3d(
            attn_windows,
            self.window_size,
            (Dp, Hp, Wp)
            )
        shifted_x = shifted_x[:, :D, :H, :W, :].contiguous()

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        x = x + self.drop_path1(self.norm1(self._attn(x)))
        x = x.reshape(B, -1, C)
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = x.reshape(B, D, H, W, C)
        return x
    

class TSwinTransformer(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 4,
            in_channels: int = 3,
            num_classes: int = 2,
            embed_dim: int = 96,
            num_frames: int = 3,
            tubelet_size: int = 1,
            temporal_encoding: _temporal_encoding_type = "doy",
            trainable_temporal_encoder: bool = False,
            num_temporal_heads: int = 4,
            depths: tuple[int, ...] = (2, 2, 6, 2),
            num_heads: tuple[int, ...] = (3, 6, 12, 24),
            window_size: int = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            patch_norm: bool = True,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            global_pool: Literal["", "avg"] = 'avg',
            weight_init: Literal["jax", "jax_nlhb", "moco", "skip", ""] = '',
            head_layer: Callable[..., nn.Module] = ClassifierHead,
            **kwargs
            ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.example_input_array = torch.randn(
            1,
            in_channels,
            num_frames,
            img_size,
            img_size
            )

        self.output_fmt = "NHWC" # TODO: get value from variable
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            output_fmt=Format3D.NDHWC
            )
        
        self.temporal_embed = TemporalEncoder(
            embed_dim=embed_dim,
            encoding_type=temporal_encoding,
            trainable_scale=trainable_temporal_encoder
            )
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        grid_size = self.patch_embed.grid_size
        self.tempo_layer = nn.Sequential(
            SwinTransformerBlock3D(
                dim=embed_dim,
                input_resolution=grid_size,
                num_heads=num_temporal_heads,
                window_size=(1, window_size, window_size),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                #drop_path=dpr[i],
                norm_layer=norm_layer
                )
            )
        self.tempo_cat = nn.Sequential(
            nn.Linear(grid_size[0] * embed_dim, grid_size[0] * embed_dim),
            nn.LayerNorm(grid_size[0] * embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(grid_size[0] * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.num_layers = len(depths)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        self.blocks = nn.Sequential(*[
            SwinTransformerV2Stage(
                dim=int(embed_dim * 2**i),
                input_resolution=(
                    grid_size[1] // 2**i, # type: ignore
                    grid_size[2] // 2**i # type: ignore
                    ),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downscale=True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
                ) for i in range(self.num_layers)
            ])
        
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.norm = norm_layer(self.num_features)
        self.head = head_layer(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt
            )
        
        if weight_init != 'skip':
            self.init_weights(weight_init)

    @property
    def backbone_params(self):
        return itertools.chain(
            self.patch_embed.parameters(),
            self.pos_drop.parameters(),
            self.blocks.parameters(),
            self.norm.parameters()
            )

    @property
    def head_params(self):
        return self.head.parameters()

    @torch.jit.unused
    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""]):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -np.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)
    
    def forward_backbone(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> torch.Tensor:
        x = self.patch_embed(x)
        if temporal_coords is not None:
            temporal_embed = self.temporal_embed(temporal_coords)
            B, T, C = temporal_embed.shape
            temporal_embed = temporal_embed.view(B, T, 1, 1, C)
            x = x + temporal_embed
        x = self.tempo_layer(x)
        x = x.permute(0, 2, 3, 1, 4)
        x = x.flatten(3)
        x = self.tempo_cat(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.head(x, pre_logits=pre_logits)
        return x

    def forward(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> torch.Tensor:
        x = self.forward_backbone(x, temporal_coords)
        x = self.forward_head(x)
        return x


class PatchExpand(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_scale: int = 2,
            norm_layer: Callable = nn.LayerNorm
            ):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.out_dim = dim * dim_scale
        self.norm = norm_layer(dim // dim_scale)
        self.expansion = nn.Linear(dim, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expansion(x)
        B, H, W, C = x.shape
        x = x.reshape(
            B,
            H,
            W,
            self.dim_scale,
            self.dim_scale,
            C // self.dim_scale**2
            )
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(
            B,
            H * self.dim_scale,
            W * self.dim_scale,
            C // self.dim_scale**2
            )
        x = self.norm(x)
        return x


class SwinTransformerV2StageUp(nn.Module):
    def __init__(
            self,
            dim: int,
            input_resolution: int | tuple[int, int],
            depth: int,
            num_heads: int,
            window_size: int,
            upscale: bool = True,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float | list[float] = 0.,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm
            ):
        super().__init__()
        self.blocks = nn.Sequential(*[
            SwinTransformerV2Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer # type: ignore
                ) for i in range(depth)
            ])
        self.expansion = PatchExpand(
            dim,
            dim_scale=2,
            norm_layer=norm_layer
            ) if upscale else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.expansion(x)
        return x


class SwinEncoderDecoder(TSwinTransformer):
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 4,
            in_channels: int = 3,
            num_classes: int = 2,
            embed_dim: int = 96,
            num_frames: int = 3,
            tubelet_size: int = 1,
            temporal_encoding: _temporal_encoding_type = "doy",
            trainable_temporal_encoder: bool = False,
            num_temporal_heads: int = 4,
            depths: tuple[int, ...] = (2, 2, 6, 2),
            num_heads: tuple[int, ...] = (3, 6, 12, 24),
            window_size: int = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            patch_norm: bool = True,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            weight_init: Literal["jax", "jax_nlhb", "moco", "skip", ""] = '',
            **kwargs
            ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            temporal_encoding=temporal_encoding,
            trainable_temporal_encoder=trainable_temporal_encoder,
            num_temporal_heads=num_temporal_heads,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            norm_layer=norm_layer,
            weight_init='skip',
            **kwargs
            )
        self.unet = UnetDecoder(self.num_features, self.num_layers)
        self.norm_up = nn.BatchNorm2d(embed_dim)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=2,
                stride=2,
                padding=0
                ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0
                )
            )
        
        if weight_init != 'skip':
            self.init_weights(weight_init)

    def forward_down(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        if temporal_coords is not None:
            temporal_embed = self.temporal_embed(temporal_coords)
            B, T, C = temporal_embed.shape
            temporal_embed = temporal_embed.view(B, T, 1, 1, C)
            x = x + temporal_embed
        x = self.tempo_layer(x)
        x = x.permute(0, 2, 3, 1, 4)
        x = x.flatten(3)
        x = self.tempo_cat(x)
        x = self.pos_drop(x)
        interms = []
        for i in range(self.num_layers):
            interms.append(x.permute(0, 3, 1, 2))
            x = self.blocks[i](x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return interms + [x]
    
    def forward_up(self, features: list[torch.Tensor]) -> torch.Tensor:
        x = self.unet(features)
        x = self.norm_up(x)
        x = self.head(x)
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=True
            )
        return x

    def forward(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> torch.Tensor:
        features = self.forward_down(x, temporal_coords)
        x = self.forward_up(features)
        return x


class FinalExpandHead(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            patch_size: int,
            num_classes: int,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm
            ):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = embed_dim // patch_size

        self.expand = PatchExpand(embed_dim, patch_size, norm_layer)
        self.conv = nn.Conv2d(
            self.out_dim,
            num_classes,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return x


class TSwinUnet(TSwinTransformer):
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 4,
            in_channels: int = 3,
            num_classes: int = 2,
            embed_dim: int = 96,
            num_frames: int = 3,
            tubelet_size: int = 1,
            temporal_encoding: _temporal_encoding_type = "doy",
            trainable_temporal_encoder: bool = False,
            num_temporal_heads: int = 4,
            depths: tuple[int, ...] = (2, 4, 6, 4),
            num_heads: tuple[int, ...] = (3, 8, 12, 24),
            window_size: int = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            patch_norm: bool = True,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            weight_init: Literal["jax", "jax_nlhb", "moco", "skip", ""] = '',
            head_layer: Callable[..., nn.Module] = FinalExpandHead,
            **kwargs
            ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            temporal_encoding=temporal_encoding,
            trainable_temporal_encoder=trainable_temporal_encoder,
            num_temporal_heads=num_temporal_heads,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            patch_norm=patch_norm,
            norm_layer=norm_layer,
            weight_init='skip',
            **kwargs
            )

        self.cat_layers = nn.ModuleList([
            nn.Sequential(
                    nn.Linear(
                        2 * int(embed_dim * 2 ** (self.num_layers - i - 1)),
                        int(1.5 * int(embed_dim * 2 ** (self.num_layers - i - 1)))
                        ),
                    nn.LayerNorm(
                        int(1.5 * int(embed_dim * 2 ** (self.num_layers - i - 1)))
                        ),
                    nn.ReLU(inplace=True),
                    nn.Linear(
                        int(1.5 * int(embed_dim * 2 ** (self.num_layers - i - 1))),
                        int(embed_dim * 2 ** (self.num_layers - i - 1))
                        ),
                    nn.LayerNorm(
                        int(embed_dim * 2 ** (self.num_layers - i - 1))
                        )
                ) for i in range(self.num_layers)
            ])
        grid_size = self.patch_embed.grid_size
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        self.up_blocks = nn.ModuleList([
            SwinTransformerV2StageUp(
                dim=int(embed_dim * 2 ** (self.num_layers - i - 1)),
                input_resolution=(
                    grid_size[1] // 2 ** (self.num_layers - i - 1),
                    grid_size[2] // 2 ** (self.num_layers - i - 1)
                    ),
                depth=depths[self.num_layers - i - 1],
                num_heads=num_heads[self.num_layers - i - 1],
                window_size=window_size,
                upscale=i < self.num_layers - 1,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[self.num_layers - i - 1],
                norm_layer=norm_layer
                ) for i in range(self.num_layers)
            ])
        
        self.num_features = int(embed_dim * 2 ** (self.num_layers))
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim)
        self.head = head_layer(embed_dim, patch_size, num_classes)

        if weight_init != 'skip':
            self.init_weights(weight_init)

    @property
    def head_params(self):
        return itertools.chain(
            self.up_blocks.parameters(),
            self.norm_up.parameters(),
            self.head.parameters()
            )

    def forward_down(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.patch_embed(x)
        if temporal_coords is not None:
            temporal_embed = self.temporal_embed(temporal_coords)
            B, T, C = temporal_embed.shape
            temporal_embed = temporal_embed.view(B, T, 1, 1, C)
            x = x + temporal_embed
        x = self.tempo_layer(x)
        x = x.permute(0, 2, 3, 1, 4)
        x = x.flatten(3)
        x = self.tempo_cat(x)
        x = self.pos_drop(x)
        interms = []
        for i in range(self.num_layers):
            interms.append(x)
            x = self.blocks[i](x)
        x = self.norm(x)
        return x, interms

    def forward_up(self, x: torch.Tensor, interms: list[torch.Tensor]) -> torch.Tensor:
        for i in range(self.num_layers):
            x = torch.cat([x, interms[self.num_layers - i - 1]], dim=-1)
            x = self.cat_layers[i](x)
            x = self.up_blocks[i](x)
        x = self.norm_up(x)
        x = self.head(x)
        return x

    def forward(
            self,
            x: torch.Tensor,
            temporal_coords: torch.Tensor | None = None,
            ) -> torch.Tensor:
        x, interm = self.forward_down(x, temporal_coords)
        x = self.forward_up(x, interm)
        return x
    