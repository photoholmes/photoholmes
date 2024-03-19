from typing import Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from torch import Tensor
from torch.nn import functional as F


def upscale_mask(
    coords: Tuple[NDArray, NDArray],
    mask: NDArray,
    target_size: Tuple[int, int],
    method: Literal[
        "linear", "nearest", "slinear", "cubic", "quintic", "pchip"
    ] = "linear",
    fill_value: Union[int, float] = 0,
) -> NDArray:
    """
    Upscale a mask to a target size.

    Args:
        coords: coordinates of the mask values
        mask: mask to upscale
        target_size: target size
        method: interpolation method
        fill_value: value to fill outside the mask

    Returns:
        NDArray: upscaled mask
    """
    X, Y = target_size
    interpolator = RegularGridInterpolator(
        coords, mask, method=method, bounds_error=False, fill_value=fill_value
    )
    target_coords = np.asarray(
        np.meshgrid(
            np.arange(0, X),
            np.arange(0, Y),
        )
    )
    return interpolator(target_coords.reshape(2, -1).T).reshape(Y, X).T


def simple_upscale_heatmap(
    heatmap: Tensor, scale_factor: Union[int, Tuple[int, int]]
) -> Tensor:
    """
    Upscales a heatmap by a specified scale factor.

    Args:
        heatmap (Tensor): A Tensor representing the heatmap to be upscaled.
                            Expected shape: [batch_size, height, width] or [height, width].
        scale_factor (Union[int, Tuple[int, int]]): An integer or a tuple of two integers
                                                    representing the scale factor for
                                                    height and width. If an integer is
                                                    provided, the same scaling is applied
                                                    to both dimensions.

    Returns:
        Tensor: A Tensor representing the upscaled heatmap. The spatial dimensions
                    of the heatmap are scaled by the specified factor(s).

    Note:
    - This function uses bilinear interpolation for upscaling.
    - If the input heatmap lacks a batch dimension, it is temporarily added and
      removed in the output.
    """
    add_batch_dim = heatmap.ndim == 2

    # Add a batch dimension if necessary
    if add_batch_dim:
        heatmap = heatmap.unsqueeze(0)

    current_height, current_width = heatmap.shape[-2], heatmap.shape[-1]

    # Determine the scaling factors for height and width
    if isinstance(scale_factor, int):
        height_scale, width_scale = scale_factor, scale_factor
    else:
        height_scale, width_scale = scale_factor

    # Calculate new size
    new_height = current_height * height_scale
    new_width = current_width * width_scale

    # Upscale using interpolate
    new_size = (new_height, new_width)
    upscaled_heatmap = F.interpolate(
        heatmap.unsqueeze(1),
        size=new_size,
    ).squeeze(1)

    # Remove the batch dimension if it was added
    if add_batch_dim:
        upscaled_heatmap = upscaled_heatmap.squeeze(0)

    return upscaled_heatmap


def resize_heatmap_with_trim_and_pad(
    heatmap: Tensor, target_size: Tuple[int, int]
) -> Tensor:
    """
    Resizes a heatmap to a specified size by trimming or padding with zeros.

    Args:
        heatmap (Tensor): A Tensor representing the heatmap to be resized.
                            Expected shape: [batch_size, height, width].
        target_size (Tuple[int, int]): A tuple representing the target size
                                        (height, width) to which the heatmap
                                        will be resized.

    Returns:
        Tensor: A Tensor representing the resized heatmap. The heatmap is trimmed
                    if the target size is smaller than the original size, or padded
                    with zeros if the target size is larger. The batch size of the
                    input is maintained.

    Note:
    - The function does not change the channel dimension if present. It operates
      only on the spatial dimensions (height and width).
    """
    current_height, current_width = heatmap.shape[-2], heatmap.shape[-1]
    target_height, target_width = target_size

    # Determine padding or trimming for height
    if current_height < target_height:
        # Pad height
        pad_height_top = (target_height - current_height) // 2
        pad_height_bottom = target_height - current_height - pad_height_top
    else:
        # Trim height
        pad_height_top = 0
        pad_height_bottom = 0
        heatmap = heatmap[..., :target_height, :]

    # Determine padding or trimming for width
    if current_width < target_width:
        # Pad width
        pad_width_left = (target_width - current_width) // 2
        pad_width_right = target_width - current_width - pad_width_left
    else:
        # Trim width
        pad_width_left = 0
        pad_width_right = 0
        heatmap = heatmap[..., :target_width]

    # Apply padding
    heatmap = F.pad(
        heatmap,
        (pad_width_left, pad_width_right, pad_height_top, pad_height_bottom),
        "constant",
        0,
    )
    return heatmap
