# Code derived from
# https://github.com/proteus1991/PSCC-Net
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
import random
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from photoholmes.methods.base import BaseTorchMethod, BenchmarkOutput
from photoholmes.utils.generic import load_yaml

from .config import PSCCNetArchConfig, PSCCNetConfig, PSCCNetWeights, pretrained_arch
from .network.detection_head import DetectionHead
from .network.NLCDetection import NLCDetection
from .network.seg_hrnet import HighResolutionNet

logger = logging.getLogger(__name__)


class PSCCNet(BaseTorchMethod):
    """
    Implementation of PSCCNet [Liu et al., 2022] method.

    The method implements an end to end neural network with multiple
    heads for both detection and localization.

    For more details and instruction to download the weights, see the
    original implementation at:
        https://github.com/proteus1991/PSCC-Net/tree/main

    To easily download the weights, you can use the script in
    scripts/download_psccnet_weights.py in the photoholmes repository.
    """

    def __init__(
        self,
        weights: PSCCNetWeights,
        arch_config: Union[PSCCNetArchConfig, Literal["pretrained"]] = "pretrained",
        device: str = "cpu",
        device_ids: Optional[List] = None,
        seed: int = 42,
        **kwargs,
    ):
        """
        Args:
            weights (PSCCNetWeights): The weights for the PSCCNet. The weights
                are expected to be a dictionary with the following keys:
                "FENet", "SegNet" and "ClsNet". The values are the paths to
                the weights.

            arch_config (PSCCNetArchConfig | "pretrained"): The architecture configuration
                for the PSSC Network. If "pretrained" is passed, the architecture from
                the paper will be used.
            device (str): Device to run the network. Default is "cuda:0".
            device_ids (Optional[List]): If multiple devices are available, pass
                the ids of the devices to use.
            crop_size: (List[int]): Size of the input image for the network.
            seed (int): Seed for reproducibility of experiments.
        """
        random.seed(seed)
        super().__init__(**kwargs)

        self.device = torch.device(device)
        self.device_ids = device_ids

        if isinstance(arch_config, str) and arch_config == "pretrained":
            arch = pretrained_arch
        else:
            arch = arch_config

        FENet = HighResolutionNet(arch, **kwargs)
        SegNet = NLCDetection(arch, arch.crop_size)
        ClsNet = DetectionHead(arch, arch.crop_size)

        FENet = self.init_network(FENet, weights.get("FENet", None))
        SegNet = self.init_network(SegNet, weights.get("SegNet", None))
        ClsNet = self.init_network(ClsNet, weights.get("ClsNet", None))

        self.FENet = FENet
        self.SegNet = SegNet
        self.ClsNet = ClsNet
        self.sm = nn.Softmax(dim=1)

        self.FENet.eval()
        self.SegNet.eval()
        self.ClsNet.eval()

        self.to_device(device)

    def init_network(
        self, net: nn.Module, weights_path: Optional[Union[str, Path]]
    ) -> nn.Module:
        """
        Initialize a subnetwork, loading it as a DataParallel module, setting it to the
        correct devices and loading the weights if provided.

        Args:
            net (nn.Module): The module to initialize
            weights (str | None): Path to the model weights. If None, the model
                uses random weights.

        Returns:
            nn.Module: The initialized module
        """
        net = net.to(self.device)
        net = nn.DataParallel(net, device_ids=self.device_ids)

        # load weights
        if weights_path is not None:
            net_state_dict = torch.load(weights_path, map_location=self.device)
            net.load_state_dict(net_state_dict)
        else:
            logger.warning(
                f"No weights provided for {net.module.__class__.__name__}. "
                "Using random weights."
            )

        return net.module.to(self.device)

    @torch.no_grad()
    def predict(  # type: ignore[override]
        self, image: torch.Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Run PSCCNet on a image. The image is expected to be in the range [0, 1].
        You can use the `psccnet_preprocessing` pipeline from
        `photoholmes.methods.psccnet` to preprocess the image.

        Args:
            image (torch.Tensor): the preprocessed input image.

        Returns:
            Tensor: The predicted heatmap
            Tensor: The detection score.
        """
        image = image.to(self.device)
        add_batch_dim = image.ndim == 3
        if add_batch_dim:
            image = image.unsqueeze(0)
        feat = self.FENet(image)

        # localization head
        heatmap = self.SegNet(feat)[0]
        heatmap = F.interpolate(
            heatmap,
            size=(image.size(2), image.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        if add_batch_dim:
            heatmap = heatmap.squeeze(0)
            heatmap = heatmap.squeeze(0)
        else:
            heatmap = heatmap.squeeze(1)

        # classification head
        pred_logit = self.ClsNet(feat)
        pred_logit = self.sm(pred_logit)[:, 1]

        return heatmap, pred_logit

    def benchmark(  # type: ignore[override]
        self, image: torch.Tensor
    ) -> BenchmarkOutput:
        """
        Benchmarks the PSCCNet method using the provided image and size.

        Args:
            image (Tensor): Input image tensor.

        Returns:
            BenchmarkOutput: Contains the heatmap and  detection and placeholder for
            detection.
        """
        heatmap, detection = self.predict(image)

        return {
            "heatmap": heatmap,
            "mask": None,
            "detection": detection,
        }

    @classmethod
    def from_config(
        cls, config: Optional[PSCCNetConfig | dict | str | Path]
    ) -> "PSCCNet":
        """
        Instantiate the model from configuration dictionary, yaml file or
        PSCCNetConfig object.

        Params:
            config: path to the yaml configuration or a dictionary with
                    the parameters for the model.
        """
        if isinstance(config, PSCCNetConfig):
            return cls(**config.__dict__)

        if isinstance(config, (str, Path)):
            config = load_yaml(config)

        if config is None:
            config = {}

        arch_config = config.get("arch_config", "pretrained")
        if isinstance(arch_config, dict):
            config["arch_config"] = PSCCNetArchConfig(**arch_config)

        return cls(**config)
