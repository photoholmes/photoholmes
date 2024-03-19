# Code derived from
# https://github.com/grip-unina/TruFor/blob/main/test_docker/src/models/cmx/builder_np_conf.py # noqa: E501
"""
Edited in September 2022
@author: fabrizio.guillaro, davide.cozzolino
"""

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from photoholmes.methods.base import BaseTorchMethod, BenchmarkOutput
from photoholmes.utils.generic import load_yaml

from .config import TruForArchConfig, TruForConfig, pretrained_arch
from .models.DnCNN import ActivationOptions, make_net
from .models.utils.init_func import init_weight
from .models.utils.layer import weighted_statistics_pooling

logger = logging.getLogger(__name__)
YELLOW_COLOR = "\033[93m"
END_COLOR = "\033[0m"


# This function is not in the `preprocessing` module as is used in the middle of the
# method's forward pass. The unormalized image is needed to compute the Noiseprint++
def preprc_imagenet_torch(x: Tensor) -> Tensor:
    """
    Normalizes an image tensor using ImageNet's mean and standard deviation.

    Args:
        x (Tensor): Input image tensor.

    Returns:
        Tensor: Normalized image tensor.
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).to(x.device)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    return x


def create_backbone(
    typ: Literal["mit_b2"], norm_layer: Type[nn.Module]
) -> Tuple[nn.Module, list[int]]:
    """
    Initializes a backbone network based on the specified type and normalization
    layer.

    Args:
        typ (Literal["mit_b2"]): Type of the backbone.
        norm_layer (Type[nn.Module]): Normalization layer type.

    Returns:
        Tuple[nn.Module, list[int]]: Backbone network and list of channels.
    """
    channels = [64, 128, 320, 512]
    if typ == "mit_b2":
        logger.info("Using backbone: Segformer-B2")
        from .models.cmx.encoders.dual_segformer import mit_b2 as backbone_

        backbone = backbone_(norm_fuse=norm_layer)
    else:
        raise NotImplementedError(f"backbone `{typ}` not implemented")
    return backbone, channels


class TruFor(BaseTorchMethod):
    """
    Trufor [Guillaro, et al. 2023] implementation.
    The method extracts both high-level and low-level features through a
    transformer-based architecture that combines the RGB image and a learned
    noise-sensitive fingerprint. The forgeries are detected as deviations from the
    expected regular pattern that characterizes a pristine image.
    """

    def __init__(
        self,
        arch_config: Union[TruForArchConfig, Literal["pretrained"]] = "pretrained",
        weights: Optional[Union[str, dict]] = None,
        use_confidence: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            arch_config (Union[TruForArchConfig, Literal["pretrained"]]): Specifies
                the architecture configuration.
            weights (Optional[Union[str, dict]]): Path to the weights file or a
                dictionary containing model weights.
            use_confidence (bool): Whether to use confidence maps to multiply the
                output heatmap in the benchmark method.
        """
        super().__init__()
        logger.warn(
            f"{YELLOW_COLOR}Trufor has a custom research only licence. "
            "See the LICENSE inside the method folder or at https://github.com/grip-unina/TruFor/blob/main/test_docker/LICENSE.txt. "  # noqa: E501
            "By continuing the use, you are agreeing to the conditions on their "
            f"license.{END_COLOR}"
        )

        if arch_config == "pretrained":
            arch_config = pretrained_arch

        self.use_confidence = use_confidence
        self.arch_config = arch_config
        self.norm_layer = nn.BatchNorm2d
        self.mods = arch_config.mods

        # import backbone and decoder
        self.backbone, self.channels = create_backbone(
            self.arch_config.backbone, self.norm_layer
        )

        if self.arch_config.confidence_backbone is not None:
            self.confidence_backbone, self.channels_conf = create_backbone(
                self.arch_config.confidence_backbone, self.norm_layer
            )
        else:
            self.confidence_backbone = None

        if self.arch_config.decoder == "MLPDecoder":
            logger.info("Using MLP Decoder")
            from .models.cmx.decoders.MLPDecoder import DecoderHead

            self.decode_head = DecoderHead(
                in_channels=self.channels,
                num_classes=self.arch_config.num_classes,
                norm_layer=self.norm_layer,
                embed_dim=self.arch_config.decoder_embed_dim,
            )

            self.decode_head_conf: Optional[nn.Module]
            if self.arch_config.confidence:
                self.decode_head_conf = DecoderHead(
                    in_channels=self.channels,
                    num_classes=1,
                    norm_layer=self.norm_layer,
                    embed_dim=self.arch_config.decoder_embed_dim,
                )
            else:
                self.decode_head_conf = None

            self.conf_detection = None
            if self.arch_config.detection is not None:
                if self.arch_config.detection is None:
                    pass

                elif self.arch_config.detection == "confpool":
                    self.conf_detection = "confpool"
                    assert self.arch_config.confidence
                    self.detection = nn.Sequential(
                        nn.Linear(in_features=8, out_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=128, out_features=1),
                    )
                else:
                    raise NotImplementedError("Detection mechanism not implemented")

        else:
            raise NotImplementedError("Decoder not implemented")

        num_levels = 17
        out_channel = 1
        npp_activations = [ActivationOptions.RELU] * (num_levels - 1) + [
            ActivationOptions.LINEAR
        ]
        self.dncnn = make_net(
            3,
            kernels=[
                3,
            ]
            * num_levels,
            features=[
                64,
            ]
            * (num_levels - 1)
            + [out_channel],
            bns=[
                False,
            ]
            + [
                True,
            ]
            * (num_levels - 2)
            + [
                False,
            ],
            acts=npp_activations,
            dilats=[
                1,
            ]
            * num_levels,
            bn_momentum=0.1,
            padding=1,
        )

        if self.arch_config.preprocess == "imagenet":  # RGB (mean and variance)
            self.prepro = preprc_imagenet_torch
        else:
            assert False

        if weights is not None:
            self.load_weights(weights)
        else:
            logger.warn("No weight file provided. Initiralizing random weights.")
            self.init_weights()

        self.to_device(device)
        self.eval()

    def init_weights(self):
        """
        Initializes weights of the decode head using Kaiming normal method.
        """
        init_weight(
            self.decode_head,
            nn.init.kaiming_normal_,
            self.norm_layer,
            self.arch_config.bn_eps,
            self.arch_config.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )

    def encode_decode(
        self, rgb: Optional[Tensor], modal_x: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Processes input RGB and modal data to produce output maps.

        Args:
            rgb (Optional[Tensor]): RGB image tensor.
            modal_x (Optional[Tensor]): Modal information tensor.

        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tensor]]: Output heatmap,
            confidence map, and detection map.
        """
        if rgb is not None:
            orisize = rgb.shape
        else:
            orisize = modal_x.shape

        # cmx
        x = self.backbone(rgb, modal_x)
        out, _ = self.decode_head(x, return_feats=True)
        out = F.interpolate(out, size=orisize[2:], mode="bilinear", align_corners=False)

        # confidence
        if self.decode_head_conf is not None:
            if self.confidence_backbone is not None:
                x_conf = self.confidence_backbone(rgb, modal_x)
            else:
                x_conf = x  # same encoder of Localization Network

            conf = self.decode_head_conf(x_conf)
            conf = F.interpolate(
                conf, size=orisize[2:], mode="bilinear", align_corners=False
            )
        else:
            conf = None

        # detection
        if self.conf_detection is not None and conf is not None:
            if self.conf_detection == "confpool":
                f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
                f2 = weighted_statistics_pooling(
                    out[:, 1:2, :, :] - out[:, 0:1, :, :], F.logsigmoid(conf)
                ).view(out.shape[0], -1)
                det = self.detection(torch.cat((f1, f2), -1))
            else:
                assert False
        else:
            det = None

        return out, conf, det

    def forward(
        self, rgb: torch.Tensor
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass of the TruFor model.

        Args:
            rgb (torch.Tensor): Input RGB image tensor.

        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]: Output
            heatmap, confidence map, detection score, and Noiseprint++ map.
        """
        # Noiseprint++ extraction
        if "NP++" in self.mods:
            modal_x = self.dncnn(rgb)
            modal_x = torch.tile(modal_x, (3, 1, 1))
        else:
            modal_x = None

        if self.prepro is not None:
            rgb = self.prepro(rgb)

        out, conf, det = self.encode_decode(rgb, modal_x)
        return out, conf, det, modal_x

    def predict(
        self, image: torch.Tensor
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Runs Trufor on an image.

        Args:
            image (torch.Tensor): input image tensor.

        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]: Output
            heatmap, confidence map, detection score, and Noiseprint++ map.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            out, conf, det, npp = self.forward(image)

        if conf is not None:
            conf = torch.squeeze(conf, 0)
            conf = torch.sigmoid(conf)[0]

        if npp is not None:
            npp = torch.squeeze(npp, 0)[0]

        if det is not None:
            det = torch.sigmoid(det)[0]

        out = torch.squeeze(out, 0)
        heatmap = F.softmax(out, dim=0)[1]
        return heatmap, conf, det, npp

    def benchmark(self, image: Tensor) -> BenchmarkOutput:
        """
        Benchmarks the TruFor method using the provided image.

        Args:
            image (Tensor): Input image tensor.

        Returns:
            BenchmarkOutput: Contains the heatmap and detection and placeholder for
            mask.
        """
        heatmap, conf, det, _ = self.predict(image)
        if self.use_confidence:
            heatmap = heatmap * conf
        return {"heatmap": heatmap, "mask": None, "detection": det}

    @classmethod
    def from_config(cls, config: Optional[TruForConfig | dict | str | Path]):
        if isinstance(config, TruForConfig):
            return cls(**config.__dict__)

        if isinstance(config, str) or isinstance(config, Path):
            config = load_yaml(str(config))
        elif config is None:
            config = {}

        trufor_config = TruForConfig(**config)

        return cls(
            arch_config=trufor_config.arch,
            weights=trufor_config.weights,
            use_confidence=trufor_config.use_confidence,
            device=trufor_config.device,
        )
