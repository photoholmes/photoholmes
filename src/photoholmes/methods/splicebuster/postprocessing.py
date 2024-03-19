# code derived from https://www.grip.unina.it/download/prog/Splicebuster/
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


def normalize_non_nan(image: NDArray) -> NDArray:
    """
    Heatmap normalization ignoring nan values.

    Args:
        image: input heatmap

    Returns:
        NDArray: normalized heatmap
    """
    if np.isnan(image).all() or np.nanmin(image) == np.nanmax(image):
        return np.zeros_like(image)
    else:
        img_max = np.nanmax(image)
        img_min = np.nanmin(image)
        normalized_image = 255 * (image - img_min) / (img_max - img_min)
        normalized_uint_img = np.nan_to_num(normalized_image, nan=0).astype(
            np.uint8
        )  # To match original implementation's processing
    return normalized_uint_img / 255


def resize_heatmap_and_pad(
    heatmap: NDArray, coords: Tuple[NDArray, NDArray], shape_out: Tuple[int, int]
) -> NDArray:
    """
    Heatmap resizing and padding with extrapolation, as the original implementation
    does.

    Args:
        heatmap (NDArray): input heatmap
        coords (Tuple[NDArray, NDArray]): coordinates of the heatmap
        shape_out (Tuple[int, int]): output shape

    Returns:
        NDArray: resized heatmap
    """
    X, Y = shape_out
    rangex, rangey = coords
    x_axis = np.arange(Y)
    y_axis = np.arange(X)
    y = interp1d(
        rangey,
        heatmap,
        axis=1,
        kind="nearest",
        fill_value="extrapolate",
        assume_sorted=True,
        bounds_error=False,
    )
    y = interp1d(
        rangex,
        y(x_axis),
        axis=0,
        kind="nearest",
        fill_value="extrapolate",
        assume_sorted=True,
        bounds_error=False,
    )
    return y(y_axis).astype(heatmap.dtype)
