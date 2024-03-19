import logging
from pathlib import Path
from typing import Dict, Union, cast

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def prune_original_weights(weights: Union[str, Path, Dict[str, Tensor]], out_path: str):
    """
    Prune the original weights from the model to match the photoholmes model
    implementation.

    Args:
        weights (Union[str, Path, Dict[str, Tensor]]): Path to the original weights.
        out_path (str): Path to save the pruned weights.
    """
    if isinstance(weights, (str, Path)):
        logger.info(f"Loading weights from {weights}")
        weights = torch.load(weights, map_location="cpu")["model"]
        weights = cast(Dict[str, Tensor], weights)

    new_weights = {}
    for k, v in weights.items():
        new_weights[k.replace("model.", "")] = v

    new_weights.pop("positional_embedding")
    new_weights.pop("ln_final.weight")
    new_weights.pop("ln_final.bias")
    new_weights.pop("token_embedding.weight")
    new_weights.pop("text_projection")
    new_weights.pop("sink_temp")

    logger.info(f"Saving weights to {out_path}")
    torch.save(new_weights, out_path)
