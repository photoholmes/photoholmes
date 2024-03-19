from typing import List, Literal, Optional, Union

from pydantic import BaseModel


class StageConfig(BaseModel):
    num_channels: List[int]
    block: Literal["BOTTLENECK", "BASIC"]
    num_blocks: List[int]
    num_branches: int
    num_modules: int
    fuse_method: Optional[Literal["SUM", "CAT"]]


class CatnetArchConfig(BaseModel):
    stage1: StageConfig
    stage2: StageConfig
    stage3: StageConfig
    stage4: StageConfig
    dc_stage3: StageConfig
    dc_stage4: StageConfig
    stage5: StageConfig
    final_conf_kernel: int
    bn_momentum: float = 0.01
    num_classes: int = 2


pretrained_arch = CatnetArchConfig(
    final_conf_kernel=1,
    stage1=StageConfig(
        num_modules=1,
        num_branches=1,
        block="BOTTLENECK",
        num_blocks=[4],
        num_channels=[64],
        fuse_method="SUM",
    ),
    stage2=StageConfig(
        num_modules=1,
        num_branches=2,
        block="BASIC",
        num_blocks=[4, 4],
        num_channels=[48, 96],
        fuse_method="SUM",
    ),
    stage3=StageConfig(
        num_modules=4,
        num_branches=3,
        block="BASIC",
        num_blocks=[4, 4, 4],
        num_channels=[48, 96, 192],
        fuse_method="SUM",
    ),
    stage4=StageConfig(
        num_modules=3,
        num_branches=4,
        block="BASIC",
        num_blocks=[4, 4, 4, 4],
        num_channels=[48, 96, 192, 384],
        fuse_method="SUM",
    ),
    dc_stage3=StageConfig(
        num_modules=3,
        num_branches=2,
        block="BASIC",
        num_blocks=[4, 4],
        num_channels=[96, 192],
        fuse_method="SUM",
    ),
    dc_stage4=StageConfig(
        num_modules=2,
        num_branches=3,
        block="BASIC",
        num_blocks=[4, 4, 4],
        num_channels=[96, 192, 384],
        fuse_method="SUM",
    ),
    stage5=StageConfig(
        num_modules=1,
        num_branches=4,
        block="BASIC",
        num_blocks=[4, 4, 4, 4],
        num_channels=[24, 48, 96, 192],
        fuse_method="SUM",
    ),
    bn_momentum=0.01,
    num_classes=2,
)


class CatnetConfig(BaseModel):
    weights: Optional[Union[str, dict]]
    arch: Union[CatnetArchConfig, Literal["pretrained"]] = "pretrained"
