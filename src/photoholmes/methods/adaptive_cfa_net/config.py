from typing import List, Literal, Optional, Union

from pydantic import BaseModel


class DirFullDilConfig(BaseModel):
    n_dir: int
    n_full: int
    n_dir_dil: int
    n_full_dil: int


class SkipDoubleDirFullDilConfig(BaseModel):
    channels_in: int
    convolutions_1: DirFullDilConfig
    convolutions_2: DirFullDilConfig


class PixelwiseConfig(BaseModel):
    conv1_in_channels: int
    conv1_out_channels: int
    conv2_out_channels: int
    conv3_out_channels: int
    conv4_out_channels: int
    kernel_size: int


class BlockwiseLayerConfig(BaseModel):
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    groups: int
    activation: Literal["Softplus", "LogSoftmax"]


class BlockwiseConfig(BaseModel):
    layers: List[BlockwiseLayerConfig]


class AdaptiveCFANetArchConfig(BaseModel):
    skip_double_dir_full_dil_config: SkipDoubleDirFullDilConfig
    pixelwise_config: PixelwiseConfig
    blockwise_config: BlockwiseConfig


pretrained_arch = AdaptiveCFANetArchConfig(
    skip_double_dir_full_dil_config=SkipDoubleDirFullDilConfig(
        channels_in=3,
        convolutions_1=DirFullDilConfig(n_dir=10, n_full=5, n_dir_dil=10, n_full_dil=5),
        convolutions_2=DirFullDilConfig(n_dir=10, n_full=5, n_dir_dil=10, n_full_dil=5),
    ),
    pixelwise_config=PixelwiseConfig(
        conv1_in_channels=103,
        conv1_out_channels=30,
        conv2_out_channels=15,
        conv3_out_channels=15,
        conv4_out_channels=30,
        kernel_size=1,
    ),
    blockwise_config=BlockwiseConfig(
        layers=[
            BlockwiseLayerConfig(
                in_channels=120,
                out_channels=180,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=30,
                activation="Softplus",
            ),
            BlockwiseLayerConfig(
                in_channels=180,
                out_channels=90,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=30,
                activation="Softplus",
            ),
            BlockwiseLayerConfig(
                in_channels=90,
                out_channels=90,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=30,
                activation="Softplus",
            ),
            BlockwiseLayerConfig(
                in_channels=90,
                out_channels=45,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                activation="Softplus",
            ),
            BlockwiseLayerConfig(
                in_channels=45,
                out_channels=45,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                activation="Softplus",
            ),
            BlockwiseLayerConfig(
                in_channels=45,
                out_channels=4,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                activation="LogSoftmax",
            ),
        ]
    ),
)


class AdaptiveCFANetConfig(BaseModel):
    weights: Optional[Union[str, dict]]
    arch: Union[AdaptiveCFANetArchConfig, Literal["pretrained"]] = "pretrained"
