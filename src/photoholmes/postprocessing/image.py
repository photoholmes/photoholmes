from typing import Any, Dict, Optional, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray
from PIL.Image import Image
from torch import Tensor

T = TypeVar("T", Tensor, NDArray)


def to_tensor_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts all the values in a dictionary to tensors.

    Args:
        input_dict (Dict[str, Any]): Dictionary to be converted to tensors.

    Returns:
        Dict[str, Any]: A dictionary with all the values converted to tensors.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, Tensor):
            output_dict[key] = value.float()
        elif isinstance(value, np.ndarray):
            output_dict[key] = torch.from_numpy(value).float()
        elif isinstance(value, Image):
            output_dict[key] = torch.from_numpy(np.array(value)).float()
        elif isinstance(value, (int, float)):
            output_dict[key] = torch.tensor(value).unsqueeze(0).float()
        else:
            output_dict[key] = value
    return output_dict


def to_numpy_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts all the values in a dictionary to numpy arrays.

    Args:
        input_dict (Dict[str, Any]): Dictionary to be converted to numpy arrays.

    Returns:
        Dict[str, Any]: A dictionary with all the values converted to numpy arrays.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, np.ndarray):
            output_dict[key] = value
        elif isinstance(value, Tensor):
            output_dict[key] = value.cpu().numpy()
        elif isinstance(value, Image):
            output_dict[key] = np.array(value)
        else:
            output_dict[key] = value
    return output_dict


def zero_one_range(
    value: T, min_value: Optional[float] = None, max_value: Optional[float] = None
) -> T:
    """
    Rescales the input value to the range [0, 1].

    Args:
        value (float): Value to be rescaled.
        min_value (float): Minimum value of the input tensor.
            If None, the minimum value of the tensor is used.
            Default: None
        max_value (float): Maximum value of the input tensor.
            If None, the maximum value of the tensor is used.
            Default: None


    Returns:
        float: The rescaled value.
    """
    if min_value is None:
        min_value = value.min()
    if max_value is None:
        max_value = value.max()
    value = (value - min_value) / (max_value - min_value)
    return value
