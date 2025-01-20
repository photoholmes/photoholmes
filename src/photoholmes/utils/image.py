import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import cv2 as cv
import jpegio
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

logger = logging.getLogger(__name__)


def read_image(path: str | Path) -> Tensor:
    """
    Read an image from a file and return it as a tensor.

    Args:
        path (str): The path to the image file.

    Returns:
        Tensor: The image as a tensor.
    """
    return torch.from_numpy(
        cv.cvtColor(cv.imread(str(path)), cv.COLOR_BGR2RGB).transpose(2, 0, 1)
    )


def save_image(path: str, img: Tensor | NDArray, *args):
    """
    Save an image to a file.

    Args:
        path (str): The path to the file.
        img (Tensor | NDArray): The image to save.
        *args: Additional arguments to pass to `cv.imwrite`.
    """

    if isinstance(img, Tensor):
        img_bgr = cv.cvtColor(tensor2numpy(img), cv.COLOR_RGB2BGR)
    else:
        img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(path, img_bgr, *args)


def tensor2numpy(image: Tensor) -> NDArray:
    """
    Convert a tensor to a numpy array and transpose the dimensions.

    Args:
        image (Tensor): The image to convert.

    Returns:
        NDArray: The image as a numpy array.
    """
    img = image.numpy()
    return img.transpose(1, 2, 0) if image.ndim > 2 else img


def plot(
    image: Tensor | NDArray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Function for easily plotting one image.

    Args:
        image (Tensor | NDArray): The image to plot.
        title Optional[str]: The title of the plot.
        save_path Optional[str]: The path to save the plot to.
    """
    if isinstance(image, Tensor):
        image = tensor2numpy(image)
    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis(False)
    if save_path is not None:
        plt.savefig(save_path)
        print("Figure saved at:", save_path)
    plt.show()


def plot_multiple(
    images: List[Tensor | NDArray],
    title: Optional[str] = None,
    ncols: int = 4,
    save_path: Optional[str] = None,
    titles: Optional[List[Optional[str]]] = None,
):
    """
    Function for easily plotting multiple images.

    Args:
        images (List[Tensor | NDArray]): The images to plot.
        title Optional[str]: The title of the plot.
        ncols int: The number of columns in the plot.
        save_path Optional[str]: The path to save the plot to.
        titles Optional[List[Optional[str]]]: The titles of the images.
    """
    N = len(images)
    nrows = np.ceil(N / ncols).astype(int)
    if titles is None:
        titles = [None] * len(images)  # type: ignore
    if nrows > 1:
        _, ax = plt.subplots(nrows, ncols)
        for n, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = tensor2numpy(img)
            i = n // ncols
            j = n % ncols
            ax[i, j].imshow(img)
            ax[i, j].set_title(titles[n])  # type: ignore
            ax[i, j].set_axis_off()
    else:
        fig, ax = plt.subplots(1, N)
        for n, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = tensor2numpy(img)
            ax[n].imshow(img)
            ax[n].set_title(titles[n])  # type: ignore
            ax[n].set_axis_off()
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print("Figure saved at:", save_path)
    plt.show()


def overlay_mask(img: NDArray, heatmap: NDArray) -> NDArray:
    """
    Overlay a heatmap on an image.

    Args:
        img (NDArray): The image.
        heatmap (NDArray): The heatmap.

    Returns:
        NDArray: The image with the heatmap overlayed.
    """
    # Normalize the heatmap to 0-255 and convert to 8-bit unsigned integer
    heatmap_normalized = cv.normalize(
        heatmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX
    )
    heatmap_uint8 = np.uint8(heatmap_normalized)

    # Apply the color map
    heatmap_img = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)

    # Superimpose the heatmap on the image
    super_imposed_img = cv.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

    # Convert superimposed image from BGR to RGB for plotting
    super_imposed_img_rgb = cv.cvtColor(super_imposed_img, cv.COLOR_BGR2RGB)
    return super_imposed_img_rgb


def read_jpeg_data(
    image_path: str,
    num_dct_channels: Optional[int] = None,
    all_quant_tables: bool = False,
    suppress_not_jpeg_warning: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Reads image from path and returns DCT coefficient matrix for each channel and the
    quantization matrixes. If image is in jpeg format, it decodes the DCT stream and
    returns it. Otherwise, the image is saved into a temporary jpeg file and then the
    DCT stream is decoded.

    Args:
        image_path (str): Path to the image.
        num_dct_channels (int, optional): Number of channels to read from the DCT stream.
            Defaults to None.
        all_quant_tables (bool, optional): If True, returns all quantization tables.
            Defaults to False.
        suppress_not_jpeg_warning (bool, optional): If True, suppresses the warning
            when the image is not in JPEG format. Defaults to False.

    Returns:
        Tuple[Tensor, Tensor]: DCT coefficients and quantization tables.
    """

    if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        jpeg = jpegio.read(image_path)
    else:
        if not suppress_not_jpeg_warning:
            logger.warning(
                "Image is not in JPEG format. An approximation will be loaded by "
                "compressing the image in quality 100."
            )
        temp = NamedTemporaryFile(suffix=".jpg")
        img = read_image(image_path)
        save_image(temp.name, img, [cv.IMWRITE_JPEG_QUALITY, 100])
        jpeg = jpegio.read(temp.name)

    return torch.tensor(
        _DCT_from_jpeg(jpeg, num_channels=num_dct_channels)
    ), torch.tensor(np.array(_qtables_from_jpeg(jpeg, all=all_quant_tables)))


def _qtables_from_jpeg(jpeg: jpegio.DecompressedJpeg, all: bool = False) -> NDArray:
    """
    Gets the quantization tables from a JPEG image.

    Args:
        jpeg (jpegio.DecompressedJpeg): The decompressed JPEG image.
        all (bool, optional): If True, returns all quantization tables.
            Defaults to False.

    Returns:
        NDArray: The quantization tables.
    """
    if all:
        return np.array(
            [jpeg.quant_tables[i].copy() for i in range(len(jpeg.quant_tables))]
        )
    else:
        return np.array(jpeg.quant_tables[0].copy())


def _DCT_from_jpeg(
    jpeg: jpegio.DecompressedJpeg, num_channels: Optional[int] = None
) -> NDArray:
    """
    Gets the DCT coefficients from a JPEG image.

    Args:
        jpeg (jpegio.DecompressedJpeg): The decompressed JPEG image.
        num_channels (int, optional): Number of channels to read from the DCT stream.
            Defaults to None.

    Returns:
        NDArray: The DCT coefficients.

    Note: Code derived from https://github.com/mjkwon2021/CAT-Net.git.
    """
    if num_channels is None:
        num_channels = len(jpeg.coef_arrays)
    ci = jpeg.comp_info

    sampling_factors = np.array(
        [[ci[i].v_samp_factor, ci[i].h_samp_factor] for i in range(num_channels)]
    )
    if num_channels == 3:
        if (sampling_factors[:, 0] == sampling_factors[0, 0]).all():
            sampling_factors[:, 0] = 2
        if (sampling_factors[:, 1] == sampling_factors[0, 1]).all():
            sampling_factors[:, 1] = 2
    else:
        sampling_factors[0, :] = 2

    dct_shape = jpeg.coef_arrays[0].shape
    DCT_coef = np.empty((num_channels, *dct_shape))

    for i in range(num_channels):
        r, c = jpeg.coef_arrays[i].shape
        block_coefs = (
            jpeg.coef_arrays[i].reshape(r // 8, 8, c // 8, 8).transpose(0, 2, 1, 3)
        )
        r_factor, c_factor = 2 // sampling_factors[i][0], 2 // sampling_factors[i][1]
        channel_coefficients = np.zeros((r * r_factor, c * c_factor))
        channel_coefficient_blocks = channel_coefficients.reshape(
            r // 8, r_factor * 8, c // 8, c_factor * 8
        ).transpose(0, 2, 1, 3)
        channel_coefficient_blocks[:, :, :, :] = np.tile(
            block_coefs, (r_factor, c_factor)
        )

        DCT_coef[i, :, :] = channel_coefficients[: dct_shape[0], : dct_shape[1]]

    return DCT_coef.astype(int)
