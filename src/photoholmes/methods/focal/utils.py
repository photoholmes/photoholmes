from typing import Any, Dict, Union

import torch
import torch.nn as nn


def load_weights(
    model: nn.Module, weights: Union[str, Dict[str, Any]], device: str = "cpu"
):
    """
    Load weights into a model.

    Args:
        model (nn.Module): model to load the weights into.
        weights (str | dict): path to the weights file or the weights themselves.
        device (str): device to run the model on.
    """
    if isinstance(weights, str):
        weights_ = torch.load(weights, map_location=torch.device(device))
    else:
        weights_ = weights

    model.load_state_dict(weights_)
