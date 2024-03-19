from typing import Literal, Optional, Union

from pydantic import BaseModel


class ClipModelConfig(BaseModel):
    vision: Literal["resnet50"]
    text: Literal["distilbert"]
    pooling: Literal["cls", "mean"]


class EXIFAsLanguageArchConfig(BaseModel):
    clip_model: ClipModelConfig
    patch_size: int
    num_per_dim: int
    feat_batch_size: int
    pred_batch_size: int
    ms_window: int
    ms_iter: int


pretrained_arch = EXIFAsLanguageArchConfig(
    clip_model=ClipModelConfig(vision="resnet50", text="distilbert", pooling="mean"),
    patch_size=128,
    num_per_dim=30,
    feat_batch_size=32,
    pred_batch_size=1024,
    ms_window=10,
    ms_iter=5,
)


class EXIFAsLanguageConfig(BaseModel):
    weights: Optional[Union[str, dict]]
    arch_config: Union[EXIFAsLanguageArchConfig, Literal["pretrained"]] = "pretrained"
    device: str = "cpu"
    seed: int = 44
