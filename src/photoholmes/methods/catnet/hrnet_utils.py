from __future__ import absolute_import, division, print_function

import logging
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch.nn as nn
import torch.nn.functional as F

from photoholmes.methods.catnet.config import StageConfig

logger = logging.getLogger(__name__)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        bn_momentum: float = 0.01,
        downsample=None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        bn_momentum: float = 0.01,
        downsample=None,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


blocks_dict: Dict[str, Union[Type[BasicBlock], Type[Bottleneck]]] = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck,
}


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        blocks: Union[Type[BasicBlock], Type[Bottleneck]],
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        fuse_method: Optional[Literal["SUM", "CAT"]],
        bn_momentum: float = 0.01,
        multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.bn_momentum = bn_momentum

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers: nn.ModuleList = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=self.bn_momentum,
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample=downsample,
                bn_momentum=self.bn_momentum,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    bn_momentum=self.bn_momentum,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return nn.ModuleList()

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_inchannels[i], momentum=self.bn_momentum
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3,
                                        momentum=self.bn_momentum,
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3,
                                        momentum=self.bn_momentum,
                                    ),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])  # type: ignore
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),  # type: ignore
                        size=[height_output, width_output],
                        mode="bilinear",
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])  # type: ignore
            x_fuse.append(self.relu(y))

        return x_fuse


def make_transition_layer(num_channels_pre_layer, num_channels_cur_layer, bn_momentum):
    num_branches_cur = len(num_channels_cur_layer)
    num_branches_pre = len(num_channels_pre_layer)

    transition_layers = []
    for i in range(num_branches_cur):
        if i < num_branches_pre:
            if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                transition_layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            num_channels_pre_layer[i],
                            num_channels_cur_layer[i],
                            3,
                            1,
                            1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=bn_momentum),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                transition_layers.append(None)
        else:
            conv3x3s = []
            for j in range(i + 1 - num_branches_pre):
                inchannels = num_channels_pre_layer[-1]
                outchannels = (
                    num_channels_cur_layer[i]
                    if j == i - num_branches_pre
                    else inchannels
                )
                conv3x3s.append(
                    nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=bn_momentum),
                        nn.ReLU(inplace=True),
                    )
                )
            transition_layers.append(nn.Sequential(*conv3x3s))

    return nn.ModuleList(transition_layers)


def make_layer(
    block: Union[Type[BasicBlock], Type[Bottleneck]],
    inplanes: int,
    planes: int,
    blocks: int,
    bn_momentum: float,
    stride: int = 1,
) -> nn.Module:
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
            nn.BatchNorm2d(planes * block.expansion, momentum=bn_momentum),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            downsample=downsample,
            bn_momentum=bn_momentum,
        )
    )
    inplanes = planes * block.expansion
    for _i in range(1, blocks):
        layers.append(block(inplanes, planes, bn_momentum=bn_momentum))

    return nn.Sequential(*layers)


def make_stage(
    layer_config: StageConfig,
    num_inchannels: List[int],
    bn_momentum: float,
    multi_scale_output: bool = True,
) -> Tuple[nn.Module, List[int]]:
    num_modules = layer_config.num_modules
    num_branches = layer_config.num_branches
    num_blocks = layer_config.num_blocks
    num_channels = layer_config.num_channels
    block = blocks_dict[layer_config.block]
    fuse_method = layer_config.fuse_method

    modules = []
    for i in range(num_modules):
        # multi_scale_output is only used last module
        if not multi_scale_output and i == num_modules - 1:
            reset_multi_scale_output = False
        else:
            reset_multi_scale_output = True
        modules.append(
            HighResolutionModule(
                num_branches,
                block,
                num_blocks,
                num_inchannels,
                num_channels,
                fuse_method,
                bn_momentum=bn_momentum,
                multi_scale_output=reset_multi_scale_output,
            )
        )
        num_inchannels = modules[-1].get_num_inchannels()

    return nn.Sequential(*modules), num_inchannels
