# Code derived from
# https://github.com/qbammey/adaptive_cfa_forensics/blob/master/src/structure.py
# ------------------------------------------------------------------------------
# Written by Quentin Bammey (quentin.bammey@ens-paris-saclay.fr)
# ------------------------------------------------------------------------------

"""Internal structures used in the network."""

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from photoholmes.methods.adaptive_cfa_net.config import (
    AdaptiveCFANetArchConfig,
    AdaptiveCFANetConfig,
    pretrained_arch,
)
from photoholmes.methods.base import BaseTorchMethod, BenchmarkOutput
from photoholmes.postprocessing.resizing import (
    resize_heatmap_with_trim_and_pad,
    simple_upscale_heatmap,
)
from photoholmes.utils.generic import load_yaml

logger = logging.getLogger(__name__)


class DirFullDil(nn.Module):
    """
    Module performing directional (horizontal and vertical), full, and dilated
    convolutions. It concatenates the results of each, performing a series of
    convolutions on the input image to extract features in different orientations
    and scales. Additionally, it applies dilation to capture broader contextual
    information without losing resolution.
    """

    def __init__(self, channels_in, *n_convolutions):
        """
        Args:
            channels_in (int): Number of input channels.
            n_convolutions (Tuple[int, int, int, int]): n_dir (int): Number of
            directional convolution filters, n_full (int): Number of full convolution
            filters, n_dir_dil (int): Number of directional dilated convolution filters,
            n_full_dil (int): Number of full dilated convolution filters.
        """
        super(DirFullDil, self).__init__()
        n_dir, n_full, n_dir_dil, n_full_dil = n_convolutions
        self.h1 = nn.Conv2d(channels_in, n_dir, (1, 3))
        self.h2 = nn.Conv2d(2 * n_dir + n_full, n_dir, (1, 3))
        self.v1 = nn.Conv2d(channels_in, n_dir, (3, 1))
        self.v2 = nn.Conv2d(2 * n_dir + n_full, n_dir, (3, 1))
        self.f1 = nn.Conv2d(channels_in, n_full, 3)
        self.f2 = nn.Conv2d(2 * n_dir + n_full, n_full, 3)
        self.hd = nn.Conv2d(channels_in, n_dir_dil, (1, 3), dilation=2)
        self.vd = nn.Conv2d(channels_in, n_dir_dil, (3, 1), dilation=2)
        self.fd = nn.Conv2d(channels_in, n_full_dil, 3, dilation=2)
        self.channels_out = 2 * n_dir + n_full + 2 * n_dir_dil + n_full_dil

    def forward(self, x):
        h_d = self.hd(x)[:, :, 2:-2]
        v_d = self.vd(x)[:, :, :, 2:-2]
        f_d = self.fd(x)
        h = self.h1(x)[:, :, 1:-1]
        v = self.v1(x)[:, :, :, 1:-1]
        f = self.f1(x)
        x = F.softplus(torch.cat((h, v, f), 1))
        h = self.h2(x)[:, :, 1:-1]
        v = self.v2(x)[:, :, :, 1:-1]
        f = self.f2(x)
        return torch.cat((h_d, v_d, f_d, h, v, f), 1)


class SkipDoubleDirFullDil(nn.Module):
    """
    Module combining two DirFullDil modules with a skip connection between them.
    It allows for the input to bypass the first convolutional block, combining
    its features with the output of the first block before passing through the
    second convolutional block. This approach is meant to help the network
    preserve information from the input through the layers.
    """

    def __init__(self, channels_in, convolutions_1, convolutions_2):
        """
        Args:
            channels_in (int): Number of input channels.
            convolutions_1 (DirFullDirConfig): Configuration for the first DirFullDil
                module.
            convolutions_2 (DirFullDirConfig): Configuration for the second DirFullDil
                module.
        """
        super(SkipDoubleDirFullDil, self).__init__()
        self.conv1 = DirFullDil(channels_in, *convolutions_1)
        self.conv2 = DirFullDil(channels_in + self.conv1.channels_out, *convolutions_2)
        self.channels_out = self.conv2.channels_out
        self.padding = 4

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = torch.cat((x[:, :, 2:-2, 2:-2], x1), 1)
        x2 = self.conv2(x1)
        x2 = torch.cat((x1[:, :, 2:-2, 2:-2], x2), 1)
        return x2


class SeparateAndPermutate(nn.Module):
    """
    Rearranges the pixels of the input feature map by separating them based on
    their position in the original image grid and permutating them in four possible
    ways. This operation aims to prepare the data for pixel-wise analysis in
    subsequent layers by highlighting different spatial relationships.
    """

    def forward(self, x):
        n, C, Y, X = x.shape
        assert n == 1
        assert Y % 2 == 0
        assert X % 2 == 0
        x_00 = x[:, :, ::2, ::2]
        x_01 = x[:, :, ::2, 1::2]
        x_10 = x[:, :, 1::2, ::2]
        x_11 = x[:, :, 1::2, 1::2]

        ind = [k + C * i for k in range(C) for i in range(4)]

        xx_00 = torch.cat((x_00, x_01, x_10, x_11), 1)[:, ind]
        xx_01 = torch.cat((x_01, x_00, x_11, x_10), 1)[:, ind]
        xx_10 = torch.cat((x_10, x_11, x_00, x_01), 1)[:, ind]
        xx_11 = torch.cat((x_11, x_10, x_01, x_00), 1)[:, ind]

        x = torch.cat((xx_00, xx_01, xx_10, xx_11), 0)
        return x


class Pixelwise(nn.Module):
    """
    Applies a sequence of convolutions to the input feature map for pixel-wise
    feature extraction.
    """

    def __init__(
        self,
        channels_in: int = 103,
        conv1_out_channels: int = 30,
        conv2_out_channels: int = 15,
        conv3_out_channels: int = 15,
        conv4_out_channels: int = 30,
        kernel_size: int = 1,
    ):
        """
        Args:
            channels_in (int): Number of input channels. Default is 103.
            conv1_out_channels (int): Number of output channels for the first convolution.
                Default is 30.
            conv2_out_channels (int): Number of output channels for the second convolution.
                Default is 15.
            conv3_out_channels (int): Number of output channels for the third convolution.
                Default is 15.
            conv4_out_channels (int): Number of output channels for the fourth convolution.
                Default is 30.
            kernel_size (int): Size of the convolutional kernels. Default is 1.
        """
        super(Pixelwise, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, conv1_out_channels, kernel_size)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size)
        self.conv3 = nn.Conv2d(
            conv1_out_channels + conv2_out_channels, conv3_out_channels, kernel_size
        )
        self.conv4 = nn.Conv2d(
            conv1_out_channels + conv2_out_channels + conv3_out_channels,
            conv4_out_channels,
            kernel_size,
        )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x))
        x3 = F.leaky_relu(self.conv3(torch.cat((x, x2), 1)))
        x4 = self.conv4(torch.cat((x, x2, x3), 1))
        return x4


class AdaptiveCFANet(BaseTorchMethod):
    """
    Implements the Adaptive CFA Net architecture [Bammey, et al. 2020] for image forgery
    detection. This network applies spatial convolutions to extract pixel-wise features,
    separates grid pixels for detailed analysis, and uses block-wise convolutions
    to analyze larger image regions. It outputs a heatmap indicating the likelihood
    of forgery in different image areas.
    """

    def __init__(
        self,
        arch_config: Union[
            AdaptiveCFANetArchConfig, Literal["pretrained"]
        ] = "pretrained",
        weights: Optional[Union[str, Path, dict]] = None,
        **kwargs,
    ):
        """
        Args:
            arch_config (Union[AdaptiveCFANetArchConfig, Literal['pretrained']]):
                Configuration for the network architecture. Can be a predefined
                architecture or 'pretrained' for default settings.
            weights (Optional[Union[str, Path, dict]]): Weights to load into the model.
                Can be a file path, dictionary, or None for random initialization.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        if arch_config == "pretrained":
            arch_config = pretrained_arch

        self.arch_config = arch_config

        self.load_model(arch_config)

        if weights is not None:
            self.load_weights(weights)
        else:
            self.init_weights()

    def load_model(self, arch_config: AdaptiveCFANetArchConfig):
        """
        Initialize the network architecture using the provided configuration.

        Args:
            arch_config (AdaptiveCFANetArchConfig): Configuration for the network
                architecture.
        """
        # Initialize DirFullDil using config
        conv1_config = arch_config.skip_double_dir_full_dil_config.convolutions_1
        conv2_config = arch_config.skip_double_dir_full_dil_config.convolutions_2
        self.spatial = SkipDoubleDirFullDil(
            arch_config.skip_double_dir_full_dil_config.channels_in,
            (
                conv1_config.n_dir,
                conv1_config.n_full,
                conv1_config.n_dir_dil,
                conv1_config.n_full_dil,
            ),
            (
                conv2_config.n_dir,
                conv2_config.n_full,
                conv2_config.n_dir_dil,
                conv2_config.n_full_dil,
            ),
        )

        # Initialize Pixelwise using config
        pw_config = arch_config.pixelwise_config
        self.pixelwise = Pixelwise(
            pw_config.conv1_in_channels,
            pw_config.conv1_out_channels,
            pw_config.conv2_out_channels,
            pw_config.conv3_out_channels,
            pw_config.conv4_out_channels,
            pw_config.kernel_size,
        )

        # Initialize auxiliary using config
        self.auxiliary = nn.Sequential(
            self.spatial, self.pixelwise, nn.Conv2d(30, 4, 1), nn.LogSoftmax(dim=1)
        )

        # Initialize Blockwise using config
        self.blockwise = nn.Sequential()
        for i, layer in enumerate(arch_config.blockwise_config.layers):
            conv_layer = nn.Conv2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                groups=layer.groups,
            )
            self.blockwise.add_module(f"{2*i}", conv_layer)
            if layer.activation.lower() != "none":
                activation_fn = getattr(nn, layer.activation, None)
                if activation_fn:
                    if activation_fn == nn.LogSoftmax:
                        self.blockwise.add_module(
                            f"activation_{i}", activation_fn(dim=1)
                        )
                    else:
                        self.blockwise.add_module(f"activation_{i}", activation_fn())
                else:
                    raise ValueError(
                        f"Activation function '{layer.activation}' is not supported."
                    )
        self.grids = SeparateAndPermutate()

    def init_weights(self):
        """
        Initialize the network weights.
        """
        logger.info("=> init weights from normal distribution")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, block_size=32):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input image tensor.
            block_size (int): Size of the image blocks. Default is 32.

        Returns:
            Tensor: Output tensor.
        """
        x = self.spatial(x)
        x = self.pixelwise(x)
        x = self.grids(x)
        x = F.avg_pool2d(x, block_size // 2)
        x = self.blockwise(x)
        return x

    @torch.no_grad()
    def predict(self, image: Tensor, image_size: Tuple[int, int]) -> Tensor:
        """
        Runs method for the input image.

        Args:
            image (Tensor): Input image tensor.
            image_size (Tuple[int, int]): Original image size.

        Returns:
            Tensor: Predicted heatmap.
        """
        image = image.to(self.device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        pred = self.forward(image)
        pred = torch.exp(pred)

        pred[:, 1] = pred[torch.tensor([1, 0, 3, 2]), 1]
        pred[:, 2] = pred[torch.tensor([2, 3, 0, 1]), 2]
        pred[:, 3] = pred[torch.tensor([3, 2, 1, 0]), 3]
        pred = torch.mean(pred, dim=1)

        best_grid = torch.argmax(torch.mean(pred, dim=(1, 2)))
        authentic = torch.argmax(pred, dim=0) == best_grid
        confidence = 1 - torch.max(pred, dim=0).values
        confidence = torch.clamp(confidence, 0, 1)
        confidence[authentic] = 1
        error_map = 1 - confidence

        upscaled_heatmap = simple_upscale_heatmap(error_map, 32)
        upscaled_heatmap = resize_heatmap_with_trim_and_pad(
            upscaled_heatmap, image_size
        )
        return upscaled_heatmap

    def benchmark(self, image: Tensor, image_size: Tuple[int, int]) -> BenchmarkOutput:
        """
        Benchmarks the Adaptive CFA Net method using the provided image and size.

        Args:
            image (Tensor): Input image tensor.
            image_size (Tuple[int, int]): Original image size.

        Returns:
            BenchmarkOutput: Contains the heatmap and placeholders for mask and
            detection.
        """
        heatmap = self.predict(image, image_size)
        return {"heatmap": heatmap, "mask": None, "detection": None}

    @classmethod
    def from_config(
        cls,
        config: Optional[Union[dict, str, Path, AdaptiveCFANetConfig]],
    ):
        if isinstance(config, AdaptiveCFANetConfig):
            return cls(**config.__dict__)

        if isinstance(config, str) or isinstance(config, Path):
            config = load_yaml(str(config))
        elif config is None:
            config = {}

        adaptive_cga_net_config = AdaptiveCFANetConfig(**config)

        return cls(
            arch_config=adaptive_cga_net_config.arch,
            weights=adaptive_cga_net_config.weights,
        )
