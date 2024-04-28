# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2020
@author: davide.cozzolino
"""

import math
from enum import Enum
from typing import List, Optional, Sequence, Union

import torch.nn as nn


def conv_with_padding(
    in_planes: int,
    out_planes: int,
    kernelsize: int,
    stride: int = 1,
    dilation: int = 1,
    bias: bool = False,
    padding: Optional[int] = None,
):
    if padding is None:
        padding = kernelsize // 2
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernelsize,
        stride=stride,
        dilation=dilation,
        padding=padding,
        bias=bias,
    )


def conv_init(conv: nn.Conv2d):
    """
    Reproduces conv initialization from DnCNN
    """
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2.0 / n))


def batchnorm_init(m: nn.BatchNorm2d, kernelsize: int = 3):
    """
    Reproduces batchnorm initialization from DnCNN
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2.0 / (n)))
    m.bias.data.zero_()


class ActivationOptions(Enum):
    RELU = "relu"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    SOFTMAX = "softmax"
    LINEAR = "linear"


def make_activation(
    act: Optional[ActivationOptions],
) -> Optional[Union[nn.Module, bool]]:
    if act is None:
        return None
    elif act == ActivationOptions.RELU:
        return nn.ReLU(inplace=True)
    elif act == ActivationOptions.TANH:
        return nn.Tanh()
    elif act == ActivationOptions.LEAKY_RELU:
        return nn.LeakyReLU(inplace=True)
    elif act == ActivationOptions.SOFTMAX:
        return nn.Softmax()
    elif act == ActivationOptions.LINEAR:
        return None
    else:
        assert False


def make_net(
    nplanes_in: int,
    kernels: List[int],
    features: List[int],
    bns: List[bool],
    acts: Sequence[Union[ActivationOptions, None]],
    dilats: List[int],
    bn_momentum: float = 0.1,
    padding: Optional[int] = None,
):
    """
    :param nplanes_in: number of of input feature channels
    :param kernels: list of kernel size for convolution layers
    :param features: list of hidden layer feature channels
    :param bns: list of whether to add batchnorm layers
    :param acts: list of activations
    :param dilats: list of dilation factors
    :param bn_momentum: momentum of batchnorm
    :param padding: integer for padding (None for same padding)
    """

    depth = len(features)
    assert len(features) == len(kernels)

    layers = list()
    for i in range(0, depth):
        if i == 0:
            in_feats = nplanes_in
        else:
            in_feats = features[i - 1]

        elem = conv_with_padding(
            in_feats,
            features[i],
            kernelsize=kernels[i],
            dilation=dilats[i],
            padding=padding,
            bias=not (bns[i]),
        )
        conv_init(elem)
        layers.append(elem)

        if bns[i]:
            elem = nn.BatchNorm2d(features[i], momentum=bn_momentum)
            batchnorm_init(elem, kernelsize=kernels[i])
            layers.append(elem)

        elem = make_activation(acts[i])
        if elem is not None:
            layers.append(elem)

    return nn.Sequential(*layers)


class DnCNN(nn.Module):
    """
    Implements a DnCNN network
    """

    def __init__(
        self,
        nplanes_in: int,
        nplanes_out: int,
        nfeatures: int,
        kernel: int,
        depth: int,
        activation: ActivationOptions,
        residual: bool,
        bn: bool,
        lastact: Optional[ActivationOptions] = None,
        bn_momentum: float = 0.10,
        padding: Optional[int] = None,
    ):
        """
        :param nplanes_in: number of of input feature channels
        :param nplanes_out: number of of output feature channels
        :param features: number of of hidden layer feature channels
        :param kernel: kernel size of convolution layers
        :param depth: number of convolution layers (minimum 2)
        :param bn:  whether to add batchnorm layers
        :param residual: whether to add a residual connection from input to output
        :param bn_momentum: momentm of batchnorm
        :param padding: integer for padding
        """
        super(DnCNN, self).__init__()

        self.residual = residual
        self.nplanes_out = nplanes_out
        self.nplanes_in = nplanes_in

        kernels = [
            kernel,
        ] * depth
        features = [
            nfeatures,
        ] * (depth - 1) + [
            nplanes_out,
        ]
        bns = (
            [
                False,
            ]
            + [
                bn,
            ]
            * (depth - 2)
            + [
                False,
            ]
        )
        dilats = [
            1,
        ] * depth
        acts = [activation] * (depth - 1) + [lastact]
        self.layers = make_net(
            nplanes_in,
            kernels,
            features,
            bns,
            acts,
            dilats=dilats,
            bn_momentum=bn_momentum,
            padding=padding,
        )

    def forward(self, x):
        shortcut = x

        x = self.layers(x)

        if self.residual:
            nshortcut = min(self.nplanes_in, self.nplanes_out)
            x[:, :nshortcut, :, :] = (
                x[:, :nshortcut, :, :] + shortcut[:, :nshortcut, :, :]
            )

        return x
