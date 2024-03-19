from dataclasses import dataclass


@dataclass
class FeaturesConfig:
    block_size: int = 128
    stride: int = 8
    q: int = 2
    T: int = 1


@dataclass
class RegularImageFeaturesConfig(FeaturesConfig):
    block_size: int = 128
    stride: int = 8
    q: int = 2
    T: int = 1


@dataclass
class SmallImageFeaturesConfig(FeaturesConfig):
    block_size: int = 64
    stride: int = 4
    q: int = 2
    T: int = 1


@dataclass
class SaturationMaskConfig:
    low_th: int = 6
    high_th: int = 252
    erotion_kernel_size: int = 9
