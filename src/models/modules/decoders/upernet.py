#

import torch
from torch import nn
from torch.nn import functional as F


class PSPModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            bin_sizes: tuple[int, ...] = (1, 2, 4, 6),
            drop_rate: float = 0.1
            ):
        super().__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=bin_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                ) for bin_size in bin_sizes
            ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + out_channels * len(bin_sizes),
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False
                ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        pyramids = [x]
        for stage in self.stages:
            pyramids.append(
                F.interpolate(
                    stage(x),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True
                    )
                )
        x = torch.cat(pyramids, dim=1)
        x = self.bottleneck(x)
        return x


class FPNfuse(nn.Module):
    def __init__(
            self,
            feature_channels: tuple[int, ...] = (64, 128, 256, 512),
            out_channels: int = 256
            ):
        super().__init__()
        self.conv1x1 = nn.ModuleList([
            nn.Conv2d(
                feat_size,
                out_channels,
                kernel_size=1
                ) for feat_size in feature_channels[1:]
            ])
        self.conv3x3 = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
                ) for i in range(len(feature_channels) - 1)
            ])
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(
                out_channels * len(feature_channels),
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
                ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def _upscale_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, H, W = y.shape
        x = F.interpolate(
            x,
            size=(H, W),
            mode='bilinear',
            align_corners=True
            )
        return x + y

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        features[1:] = [conv1x1(feat) for conv1x1, feat in zip(self.conv1x1, features[1:])]
        print([f.shape for f in features])
        pn = [self._upscale_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        pn = [conv3x3(x) for conv3x3, x in zip(self.conv3x3, pn)]
        pn = list(reversed(pn))
        pn.append(features[-1])
        _, _, H, W = pn[0].shape
        pn[1:] = [
            F.interpolate(
                feature,
                size=(H, W),
                mode='bilinear',
                align_corners=True
                ) for feature in pn[1:]
        ]
        x = self.conv_fuse(torch.cat(pn, dim=1))
        return x

class UperNetDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            feature_channels: tuple[int, ...] = (64, 128, 256, 512),
            psp_bin_sizes: tuple[int, ...] = (1, 2, 4, 6),
            drop_rate: float = 0.1,
            fpn_out_channels: int = 256
            ):
        super().__init__()
        self.psp = PSPModule(
            feature_channels[-1],
            bin_sizes=psp_bin_sizes,
            drop_rate=drop_rate
            )
        self.fpn = FPNfuse(
            feature_channels,
            fpn_out_channels
            )
        self.head = nn.Sequential(
            nn.Conv2d(
                fpn_out_channels,
                fpn_out_channels,
                kernel_size=3,
                padding=1,
                bias=False
                ),
            nn.BatchNorm2d(fpn_out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Conv2d(
                fpn_out_channels,
                num_classes,
                kernel_size=1
                )
            )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        features[-1] = self.psp(features[-1])
        x = self.fpn(features)
        x = self.head(x)
        return x
