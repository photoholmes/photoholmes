from pathlib import Path
from typing import List, Literal, Tuple, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class StageConfig(BaseModel):
    num_modules: int
    num_branches: int
    num_blocks: List[int]
    num_channels: List[int]
    block: Literal["BOTTLENECK", "BASIC"]
    fuse_method: Literal["SUM"]


class PSCCNetArchConfig(BaseModel):
    stage1: StageConfig
    stage2: StageConfig
    stage3: StageConfig
    stage4: StageConfig
    stem_inplanes: int = 64
    final_conv_kernel: int = 1
    crop_size: Tuple[int, int] = (256, 256)


pretrained_arch = PSCCNetArchConfig(
    final_conv_kernel=1,
    stem_inplanes=64,
    stage1=StageConfig(
        num_modules=1,
        num_branches=1,
        num_blocks=[2],
        num_channels=[64],
        block="BOTTLENECK",
        fuse_method="SUM",
    ),
    stage2=StageConfig(
        num_modules=1,
        num_branches=2,
        num_blocks=[2, 2],
        num_channels=[18, 36],
        block="BASIC",
        fuse_method="SUM",
    ),
    stage3=StageConfig(
        num_modules=1,
        num_branches=3,
        num_blocks=[2, 2, 2],
        num_channels=[18, 36, 72],
        block="BASIC",
        fuse_method="SUM",
    ),
    stage4=StageConfig(
        num_modules=1,
        num_branches=4,
        num_blocks=[2, 2, 2, 2],
        num_channels=[18, 36, 72, 144],
        block="BASIC",
        fuse_method="SUM",
    ),
    crop_size=(256, 256),
)


class PSCCNetWeights(TypedDict):
    FENet: Union[str, Path]
    SegNet: Union[str, Path]
    ClsNet: Union[str, Path]


class PSCCNetConfig(BaseModel):
    weights: PSCCNetWeights
    arch_config: Union[PSCCNetArchConfig, Literal["pretrained"]] = "pretrained"
