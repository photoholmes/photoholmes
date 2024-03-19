from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from photoholmes.methods.psccnet.config import PSCCNetArchConfig

BN_MOMENTUM = 0.01


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DetectionHead(nn.Module):
    def __init__(self, config: PSCCNetArchConfig, crop_size: Tuple[int, int]):
        """Detection head for PSCCNet
             Args:
        config (PSCCNetArchConfig): PSCCNet architecture config.
        crop_size (List[int]): feature map crop size."""
        super(DetectionHead, self).__init__()
        self.crop_size = crop_size

        pre_stage_channels = config.stage4.num_channels

        # classification head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(
            pre_stage_channels
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 16), nn.ReLU(inplace=True), nn.Linear(16, 2)
        )

    def _make_layer(
        self,
        block: Type[Bottleneck],
        inplanes: int,
        planes: int,
        blocks: int,
        stride=1,
    ) -> nn.Module:
        """Creates a layer of blocks for the head.
        Args:
            block (nn.Module class, not instanced): block to use.
            inplanes (int): number of input channels.
            planes (int): number of output channels.
            blocks (int): number of blocks to use.
            stride (int): stride to use.
        Returns:
            nn.Module: layer of blocks.
        """
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_head(
        self, pre_stage_channels: List[int]
    ) -> Tuple[nn.ModuleList, nn.ModuleList, nn.Module]:
        head_block = Bottleneck
        head_channels = pre_stage_channels

        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block, channels, head_channels[i], 1, stride=1
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, feat: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        s1, s2, s3, s4 = feat

        if s1.shape[2:] == self.crop_size:
            pass
        else:
            s1 = F.interpolate(
                s1, size=self.crop_size, mode="bilinear", align_corners=True
            )
            s2 = F.interpolate(
                s2,
                size=[i // 2 for i in self.crop_size],
                mode="bilinear",
                align_corners=True,
            )
            s3 = F.interpolate(
                s3,
                size=[i // 4 for i in self.crop_size],
                mode="bilinear",
                align_corners=True,
            )
            s4 = F.interpolate(
                s4,
                size=[i // 8 for i in self.crop_size],
                mode="bilinear",
                align_corners=True,
            )

        y_list = [s1, s2, s3, s4]

        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)

        # average and flatten
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)

        logit = self.classifier(y)

        return logit
