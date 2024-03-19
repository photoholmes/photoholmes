import glob
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from photoholmes.datasets.base import BaseDataset
from photoholmes.utils.image import read_image, read_jpeg_data


class BaseTraceDataset(BaseDataset):
    """
    Base class for the Trace dataset.

    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── noise_exo.png, image tampered with noise using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── noise_exo.png, image tampered with noise using exomask
    │   └── mask_exo.png, exomask
    │   └── More images

    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str
    FORGED_NAME: str
    MASK_NAME: str
    IMAGE_EXTENSION: str = ".png"
    MASK_EXTENSION: str = ".png"

    def _get_paths(
        self, dataset_path: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        """
        Get the paths of the images and masks in the camera directory.

        Args:
            camera_dir (str): Path to the camera directory.
            tampered_only (bool): Whether to load only the tampered images.

        Returns:
            Tuple[List[str], List[str] | List[str | None]]: Paths of the images and
                masks.
        """
        image_paths = glob.glob(os.path.join(dataset_path, "*", self.FORGED_NAME))
        mask_paths: List[Optional[str]] = [
            self._get_mask_path(image_path) for image_path in image_paths
        ]

        if not tampered_only:
            pris_paths = glob.glob(os.path.join(dataset_path, "*", self.AUTH_NAME))

            pris_msk_paths = [None] * len(pris_paths)
            image_paths += pris_paths
            mask_paths += pris_msk_paths

        return image_paths, mask_paths

    def _get_mask_path(self, image_path: str) -> str:
        """
        Get the path of the mask for the given image path.

        Args:
            image_path (str): Path to the image.

        Returns:
            str: Path to the mask.
        """
        image_dir = os.path.dirname(image_path)
        return os.path.join(image_dir, self.MASK_NAME)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        """
        Binarize the mask.

        Args:
            mask_image (Tensor): Mask image.

        Returns:
            Tensor: Binarized mask image.
        """
        return mask_image[0, :, :] > 0

    def _get_data(self, idx: int) -> Tuple[Dict, Tensor, str]:
        """
        Return the data at the given index.

        Args:
            idx (int): Index of the item to return.

        Returns:
            Tuple[Dict, Tensor, str]: A tuple containing the data, the mask and the
                image name.
        """
        x = {}

        image_path = self.image_paths[idx]

        # image_name takes the folder and the filename without extension
        # instead of just the filename without extension
        image_name = "_".join(image_path.split("/")[-2:]).split(".")[0]

        if self.load_image_data:
            image = read_image(image_path)
            x["image"] = image
        if self.load_jpeg_data:
            dct, qtables = read_jpeg_data(image_path, suppress_not_jpeg_warning=True)
            if "dct_coefficients" in self.load:
                x["dct_coefficients"] = dct
            if "qtables" in self.load:
                x["qtables"] = qtables

        if self.mask_paths[idx] is None:
            mask = torch.zeros(image.shape[-2:], dtype=torch.bool)
        else:
            mask_im = read_image(self.mask_paths[idx])
            mask = self._binarize_mask(mask_im)

        return x, mask, image_name


class TraceNoiseExoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── noise_exo.png, image tampered with noise using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── noise_exo.png, image tampered with noise using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "noise_exo.png"
    MASK_NAME: str = "mask_exo.png"


class TraceNoiseEndoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── noise_endo.png, image tampered with noise using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── noise_endo.png, image tampered with noise using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "noise_endo.png"
    MASK_NAME: str = "mask_endo.png"


class TraceJPEGQualityExoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── jpeg_grid_exo.png, image tampered with different jpeg quality
    |       using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── jpeg_grid_exo.png, image tampered with different jpeg quality
    |       using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "jpeg_quality_exo.png"
    MASK_NAME: str = "mask_exo.png"


class TraceJPEGQualityEndoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── jpeg_grid_endo.png, image tampered with different jpeg quality
    |       using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── jpeg_grid_endo.png, image tampered with different jpeg quality
    |       using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "jpeg_quality_endo.png"
    MASK_NAME: str = "mask_endo.png"


class TraceJPEGGridExoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── jpeg_grid_exo.png, image tampered with different jpeg grids
    |       using exomask
    │   └── mask_endo.png, exomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── jpeg_grid_exo.png, image tampered with different jpeg grids
    |       using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "jpeg_grid_exo.png"
    MASK_NAME: str = "mask_exo.png"


class TraceJPEGGridEndoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── jpeg_grid_endo.png, image tampered with different jpeg grids
    |       using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── jpeg_grid_endo.png, image tampered with different jpeg grids
    |       using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "jpeg_grid_endo.png"
    MASK_NAME: str = "mask_endo.png"


class TraceHybridExoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── hybrid_select_exo.png, image tampered with a combination of different
    |       pipelines using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── hybrid_select_exo.png, image tampered with a combination of different
    |       pipelines using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "hybrid_select_exo.png"
    MASK_NAME: str = "mask_exo.png"


class TraceHybridEndoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── hybrid_select_endo.png, image tampered with a combination of different
    |       pipelines using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── hybrid_select_endo.png, image tampered with a combination of different
    |       pipelines using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "hybrid_select_endo.png"
    MASK_NAME: str = "mask_endo.png"


class TraceCFAAlgExoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── cfa_alg_exo.png, image tampered with different cfa algorithms
    |       using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── cfa_alg_exo.png, image tampered with different cfa algorithms
    |       using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "cfa_alg_exo.png"
    MASK_NAME: str = "mask_exo.png"


class TraceCFAAlgEndoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── cfa_alg_endo.png, image tampered with different cfa algorithms
    |       using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── cfa_alg_endo.png, image tampered with different cfa algorithms
    |       using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "cfa_alg_endo.png"
    MASK_NAME: str = "mask_endo.png"


class TraceCFAGridExoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── cfa_grid_exo.png, image tampered with different cfa grids
    |       using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── cfa_grid_exo.png, image tampered with different cfa grids
    |       using exomask
    │   └── mask_exo.png, exomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "cfa_grid_exo.png"
    MASK_NAME: str = "mask_exo.png"


class TraceCFAGridEndoDataset(BaseTraceDataset):
    """
    Directory structure:
    img_dir (Trace)
    ├── Name of image inherited from Raise dataset
    │   └── original.NEF, original image from Raise dataset
    │   └── cfa_grid_endo.png, image tampered with different cfa grids
    |       using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    ├── ...
    ├── Name of image inherited from Raise dataset
    |   └── original.NEF, original image from Raise dataset
    │   └── cfa_grid_endo.png, image tampered with different cfa grids
    |       using endomask
    │   └── mask_endo.png, endomask
    │   └── More images
    This dataset allows the inclusion of the original images but the authors
    of the proposed dataset do not include them in their experiments.
    """

    AUTH_NAME: str = "original.NEF"
    FORGED_NAME: str = "cfa_grid_endo.png"
    MASK_NAME: str = "mask_endo.png"
