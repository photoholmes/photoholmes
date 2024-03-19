# Code derived from
# https://github.com/mjkwon2021/CAT-Net/blob/f1716b0849eb4d94687a02c25bf97229b495bf9e/lib/models/network_CAT.py#L286  # noqa: E501
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------
"""
Modified by Myung-Joon Kwon
mjkwon2021@gmail.com
Aug 22, 2020
"""
import logging
import os
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from photoholmes.methods.base import BaseTorchMethod, BenchmarkOutput
from photoholmes.methods.catnet.config import (
    CatnetArchConfig,
    CatnetConfig,
    pretrained_arch,
)
from photoholmes.methods.catnet.hrnet_utils import (
    BasicBlock,
    blocks_dict,
    make_layer,
    make_stage,
    make_transition_layer,
)
from photoholmes.postprocessing.resizing import (
    resize_heatmap_with_trim_and_pad,
    simple_upscale_heatmap,
)
from photoholmes.utils.generic import load_yaml

logger = logging.getLogger(__name__)

YELLOW_COLOR = "\033[93m"
END_COLOR = "\033[0m"


class CatNet(BaseTorchMethod):
    """
    Implements the CAT-Net method [Kwon, et al. 2021] for image forgery localization.
    The method is an end-to-end fully convolutional neural network designed to detect
    compression artifacts in images

    """

    def __init__(
        self,
        arch_config: Union[CatnetArchConfig, Literal["pretrained"]] = "pretrained",
        weights: Optional[Union[str, Path, dict]] = None,
        **kwargs,
    ):
        """
        Args:
            arch_config (Union[CatnetArchConfig, Literal['pretrained']]):
            Configuration for the network architecture. Can be a predefined
            architecture or 'pretrained' for default settings.
            weights (Optional[Union[str, Path, dict]]): Path to the weights file
        """
        super().__init__(**kwargs)

        logger.warning(
            f"{YELLOW_COLOR} CatNet is under a license that only allows research use. "
            "You can check the license inside the method folder's or at https://github.com/mjkwon2021/CAT-Net/blob/main/README.md#licence."  # noqa: E501
            "If you use this method, you are agreeing to the terms of the license."
        )

        if arch_config == "pretrained":
            arch_config = pretrained_arch

        self.bn_momentum = arch_config.bn_momentum

        self.load_model(arch_config)

        if weights is not None:
            self.load_weights(weights)
        else:
            self.init_weights()

        self.eval()

    def load_model(self, arch_config: CatnetArchConfig):
        """
        Initialize the network architecture using the provided configuration.

        Args:
            arch_config (CatnetArchConfig): Configuration for the network architecture.
        """
        # RGB branch
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.bn_momentum)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=self.bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        self.stage1_cfg = arch_config.stage1
        num_channels = self.stage1_cfg.num_channels
        block = blocks_dict[self.stage1_cfg.block]
        num_blocks = self.stage1_cfg.num_blocks[0]
        self.layer1 = make_layer(
            block, 64, num_channels[0], num_blocks, bn_momentum=self.bn_momentum
        )
        stage1_out_channel = block.expansion * num_channels[0]

        self.stage2_cfg = arch_config.stage2
        num_channels = self.stage2_cfg.num_channels
        block = blocks_dict[self.stage2_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = make_transition_layer(
            [stage1_out_channel], num_channels, self.bn_momentum
        )
        self.stage2, pre_stage_channels = make_stage(
            self.stage2_cfg, num_channels, bn_momentum=self.bn_momentum
        )

        self.stage3_cfg = arch_config.stage3
        num_channels = self.stage3_cfg.num_channels
        block = blocks_dict[self.stage3_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = make_transition_layer(
            pre_stage_channels, num_channels, self.bn_momentum
        )
        self.stage3, pre_stage_channels = make_stage(
            self.stage3_cfg, num_channels, self.bn_momentum
        )

        self.stage4_cfg = arch_config.stage4
        num_channels = self.stage4_cfg.num_channels
        block = blocks_dict[self.stage4_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = make_transition_layer(
            pre_stage_channels, num_channels, self.bn_momentum
        )
        self.stage4, RGB_final_channels = make_stage(
            self.stage4_cfg,
            num_channels,
            bn_momentum=self.bn_momentum,
            multi_scale_output=True,
        )

        # DCT coefficient branch
        self.dc_layer0_dil = nn.Sequential(
            nn.Conv2d(
                in_channels=21,
                out_channels=64,
                kernel_size=3,
                stride=1,
                dilation=8,
                padding=8,
            ),
            nn.BatchNorm2d(64, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.dc_layer1_tail = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=4,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(4, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.dc_layer2 = make_layer(
            BasicBlock,
            inplanes=4 * 64 * 2,
            planes=96,
            blocks=4,
            bn_momentum=self.bn_momentum,
            stride=1,
        )

        self.dc_stage3_cfg = arch_config.dc_stage3
        num_channels = self.dc_stage3_cfg.num_channels
        block = blocks_dict[self.dc_stage3_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.dc_transition2 = make_transition_layer(
            [96], num_channels, self.bn_momentum
        )
        self.dc_stage3, pre_stage_channels = make_stage(
            self.dc_stage3_cfg, num_channels, bn_momentum=self.bn_momentum
        )

        self.dc_stage4_cfg = arch_config.dc_stage4
        num_channels = self.dc_stage4_cfg.num_channels
        block = blocks_dict[self.dc_stage4_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.dc_transition3 = make_transition_layer(
            pre_stage_channels, num_channels, self.bn_momentum
        )
        self.dc_stage4, DC_final_stage_channels = make_stage(
            self.dc_stage4_cfg,
            num_channels,
            bn_momentum=self.bn_momentum,
            multi_scale_output=True,
        )

        DC_final_stage_channels.insert(0, 0)  # to match # branches

        # stage 5
        self.stage5_cfg = arch_config.stage5
        num_channels = self.stage5_cfg.num_channels
        block = blocks_dict[self.stage5_cfg.block]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition4 = make_transition_layer(
            [i + j for (i, j) in zip(RGB_final_channels, DC_final_stage_channels)],
            num_channels,
            self.bn_momentum,
        )
        self.stage5, pre_stage_channels = make_stage(
            self.stage5_cfg, num_channels, bn_momentum=self.bn_momentum
        )

        last_inp_channels = sum(pre_stage_channels)
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(last_inp_channels, momentum=self.bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=arch_config.num_classes,
                kernel_size=arch_config.final_conf_kernel,
                stride=1,
                padding=1 if arch_config.final_conf_kernel == 3 else 0,
            ),
        )

    def forward(self, x, qtable):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor. The first 3 channels are the RGB image, and the
                remaining channels are the DCT coefficients.
            qtable (Tensor): Quantization table for the DCT coefficients.

        Returns:
            Tensor: Output of the network.
        """
        RGB, DCTcoef = x[:, :3, :, :], x[:, 3:, :, :]

        # RGB Stream
        x = self.conv1(RGB)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg.num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg.num_branches):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.num_branches):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        RGB_list = self.stage4(x_list)

        # DCT Stream
        x = self.dc_layer0_dil(DCTcoef)
        x = self.dc_layer1_tail(x)
        B, C, H, W = x.shape
        x0 = (
            x.reshape(B, C, H // 8, 8, W // 8, 8)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(B, 64 * C, H // 8, W // 8)
        )  # [B, 256, 32, 32]
        x_temp = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(
            0, 1, 3, 5, 2, 4
        )  # [B, C, 8, 8, 32, 32]
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, C, 8, 8, 32, 32]
        x1 = xq_temp.reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x = torch.cat([x0, x1], dim=1)
        x = self.dc_layer2(x)  # x.shape = torch.Size([1, 96, 64, 64])

        x_list = []
        for i in range(self.dc_stage3_cfg.num_branches):
            if self.dc_transition2[i] is not None:
                x_list.append(self.dc_transition2[i](x))
            else:
                x_list.append(x)
        y_list = self.dc_stage3(x_list)

        x_list = []
        for i in range(self.dc_stage4_cfg.num_branches):
            if self.dc_transition3[i] is not None:
                x_list.append(self.dc_transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        DC_list = self.dc_stage4(x_list)

        # stage 5
        x = [
            torch.cat([RGB_list[i + 1], DC_list[i]], 1)
            for i in range(self.stage5_cfg.num_branches - 1)
        ]
        x.insert(0, RGB_list[0])
        x_list = []
        for i in range(self.stage5_cfg.num_branches):
            if self.transition4[i] is not None:
                x_list.append(self.transition4[i](x[i]))
            else:
                x_list.append(x[i])
        x = self.stage5(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode="bilinear")
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode="bilinear")
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode="bilinear")

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)

        return x

    def init_weights(
        self,
        pretrained_rgb="",
        pretrained_dct="",
    ):
        """
        Initialize the weights of the network.

        Args:
            pretrained_rgb (str): Path to the pretrained weights for the RGB stream.
            pretrained_dct (str): Path to the pretrained weights for the DCT stream.
        """

        logger.info("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained_rgb):
            loaded_dict = torch.load(pretrained_rgb, map_location=self.device)
            model_dict = self.state_dict()
            loaded_dict = {
                k: v
                for k, v in loaded_dict.items()
                if k in model_dict.keys() and not k.startswith("lost_layer.")
            }  # RGB weight
            logger.info(
                "=> (RGB) loading pretrained model {} ({})".format(
                    pretrained_rgb, len(loaded_dict)
                )
            )
            model_dict.update(loaded_dict)
            self.load_state_dict(model_dict)
        else:
            logger.warning("=> Cannot load pretrained RGB")
        if os.path.isfile(pretrained_dct):
            loaded_dict = torch.load(pretrained_dct, map_location=self.device)[
                "state_dict"
            ]
            model_dict = self.state_dict()
            loaded_dict = {
                k: v for k, v in loaded_dict.items() if k in model_dict.keys()
            }
            loaded_dict = {
                k: v for k, v in loaded_dict.items() if not k.startswith("last_layer")
            }
            logger.info(
                "=> (DCT) loading pretrained model {} ({})".format(
                    pretrained_dct, len(loaded_dict)
                )
            )
            model_dict.update(loaded_dict)
            self.load_state_dict(model_dict)
        else:
            logger.warning("=> Cannot load pretrained DCT")

    @torch.no_grad()
    def predict(
        self, x: Tensor, qtable: Tensor, image_size: Tuple[int, int]
    ) -> Tuple[Tensor, Tensor]:
        """
        Runs method for input image.

        Args:
            x: Preprocessed input and dct coefficients. Use catnet_preprocessing.
            qtable: Quantization table
            image_size: Original image size

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing the heatmaps for authentic and
                tampered regions.
        """
        x, qtable = x.to(self.device), qtable.to(self.device)
        add_batch_dim = x.ndim == 3
        if add_batch_dim:
            x = x.unsqueeze(0)

        pred = self.forward(x, qtable)
        pred = F.softmax(pred, dim=1)
        pred_tampered = pred[:, 1]
        pred_authentic = pred[:, 0]

        # Original implementation simply upscales the heatmap to match the
        # original image. We use a more sophisticated method to upscale the
        # that "undos" the previous trimming.
        pred_authentic = simple_upscale_heatmap(pred_authentic, 4)
        pred_tampered = simple_upscale_heatmap(pred_tampered, 4)

        pred_authentic = resize_heatmap_with_trim_and_pad(pred_authentic, image_size)
        pred_tampered = resize_heatmap_with_trim_and_pad(pred_tampered, image_size)

        if add_batch_dim:
            pred_tampered = pred_tampered.squeeze(0)
            pred_authentic = pred_authentic.squeeze(0)

        return pred_tampered, pred_authentic

    def benchmark(
        self, x: Tensor, qtable: Tensor, image_size: Tuple[int, int]
    ) -> BenchmarkOutput:
        """
        Benchmarks the CatNet method using the provided image, qtables and size.

        Args:
            x (Tensor): Input image.
            qtable (Tensor): Quantization table for the DCT coefficients.
            image_size (Tuple[int, int]): Original image size.

        Returns:
            BenchmarkOutput: Contains the heatmap and placeholders for mask and
            detection.
        """
        heatmap, _ = self.predict(x, qtable, image_size)
        return {"heatmap": heatmap, "mask": None, "detection": None}

    @classmethod
    def from_config(
        cls,
        config: Optional[CatnetConfig | dict | str | Path],
    ):
        if isinstance(config, CatnetConfig):
            return cls(**config.__dict__)

        if isinstance(config, str) or isinstance(config, Path):
            config = load_yaml(str(config))
        elif config is None:
            config = {}

        catnet_config = CatnetConfig(**config)

        return cls(arch_config=catnet_config.arch, weights=catnet_config.weights)
