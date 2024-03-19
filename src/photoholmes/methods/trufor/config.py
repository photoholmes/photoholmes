from typing import Literal, Optional, Sequence, Union

from pydantic import BaseModel


class TruForArchConfig(BaseModel):
    backbone: Literal["mit_b2"]
    decoder: str
    num_classes: int
    decoder_embed_dim: int
    preprocess: str
    bn_eps: float
    bn_momentum: float
    detection: Optional[str]
    confidence: bool
    mods: Sequence[Literal["NP++", "RGB"]]

    confidence_backbone: Optional[Literal["mit_b2"]]


pretrained_arch = TruForArchConfig(
    backbone="mit_b2",
    decoder="MLPDecoder",
    num_classes=2,
    decoder_embed_dim=512,
    preprocess="imagenet",
    bn_eps=0.001,
    bn_momentum=0.01,
    detection="confpool",
    confidence=True,
    mods=["NP++", "RGB"],
    confidence_backbone=None,
)


class TruForConfig(BaseModel):
    arch: Union[TruForArchConfig, Literal["pretrained"]] = "pretrained"
    weights: Optional[Union[str, dict]]
    use_confidence: bool = True
    device: str = "cpu"
