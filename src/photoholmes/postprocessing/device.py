from typing import Any, Dict, Union

import torch
from torch import Tensor


def to_device_dict(
    input_dict: Dict[str, Any], device: Union[str, torch.device]
) -> Dict[str, Any]:
    """
    Moves all the values in a dictionary to the specified device.

    Args:
        input_dict (Dict[str, Any]): Dictionary to be moved to the specified device.
        device (Dict[str, Any]): Device to move the dictionary to.

    Returns:
        Dict[str,Any]: A dictionary with all the values moved to the specified device.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, Tensor):
            output_dict[key] = value.to(device, dtype=torch.float32)
        else:
            output_dict[key] = value
    return output_dict
