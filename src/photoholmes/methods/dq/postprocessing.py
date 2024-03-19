from typing import Tuple

import torch
from numpy.typing import NDArray
from torch import Tensor

from photoholmes.postprocessing.resizing import (
    resize_heatmap_with_trim_and_pad,
    simple_upscale_heatmap,
)


def DQ_postprocessing(
    BPPM_upsampled: NDArray, original_image_size: Tuple[int, int]
) -> Tensor:
    """
    Postprocessing for the DQ method.

    Args:
        BPPM_upsampled: Predicted heatmap.
        original_image_size: Size of the original image.

    Returns:
        The postprocessed heatmap.
    """
    BPPM_upsampled_tensor = torch.from_numpy(BPPM_upsampled)
    BPPM_upsampled_tensor = simple_upscale_heatmap(BPPM_upsampled_tensor, 8)
    BPPM_upsampled_tensor = resize_heatmap_with_trim_and_pad(
        BPPM_upsampled_tensor, original_image_size
    )

    return BPPM_upsampled_tensor
