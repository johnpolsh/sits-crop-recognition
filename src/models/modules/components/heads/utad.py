#

import torch
from torch import nn
from typing import Optional


def expand_temporal_feature(
        x: torch.Tensor,
        grid_size: tuple[int, ...],
        embed_dim: Optional[int] = None
        ) -> torch.Tensor:
    B, N, L = x.shape
    seq_len = grid_size[0] * grid_size[1] * grid_size[2]
    assert N == seq_len, f"{N=} must be equal to {seq_len=}"
    if embed_dim is not None:
        assert L == embed_dim, f"{L=} must be equal to {embed_dim=}"
    
    x = x.transpose(1, 2)
    x = x.reshape(B, L, *grid_size)
    return x


class TemporalSelfAttention(nn.Module): # TODO
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_frames: int
            ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_frames = num_frames

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        assert L == self.num_frames, f"{L=} must be equal to {self.num_frames=}"

        x = x.transpose(0, 1)
        x = self.attn(x, x, x)[0]
        x = x.transpose(0, 1)
        return x


class DownconvBlock(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chans: int
            ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_chans,
                out_chans,
                kernel_size=3,
                stride=1,
                padding=1
                ),
            nn.BatchNorm2d(out_chans),
            nn.ReLU()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class UpconvBlock(nn.Module):
    def __init__(
            self,
            in_chans: int,
            out_chans: int
            ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans,
                out_chans,
                kernel_size=2,
                stride=2
                ),
            nn.BatchNorm2d(out_chans),
            nn.ReLU()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x



class UTADecoder(nn.Module):
    def __init__(self):
        super().__init__()
