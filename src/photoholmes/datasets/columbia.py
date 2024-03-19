import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .base import BaseDataset


class ColumbiaDataset(BaseDataset):
    """
    Class for the Columbia Uncompressed Image Splicing Detection dataset.

    Directory structure:
    img_dir (Columbia Uncompressed Image Splicing Detection)
    ├── 4cam_auth
    │   ├── [images in TIF]
    ├── 4cam_splc
    │   ├── [images in TIF]
    |   └── edgemask
    |       └── [masks in JPG]
    └── README.txt
    """

    TAMP_DIR: str = "4cam_splc"
    AUTH_DIR: str = "4cam_auth"
    MASKS_DIR: str = "4cam_splc/edgemask"
    IMAGE_EXTENSION: str = ".tif"
    MASK_EXTENSION: str = ".jpg"
    TAMPERED_COLOR_INDEX: int = 1  # Green

    def _get_paths(
        self, dataset_path: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        """
        Get the paths of the images and masks in the dataset.

        Args:
            dataset_path (str): Path to the dataset.
            tampered_only (bool): Whether to load only the tampered images.

        Returns:
            Tuple[List[str], List[str] | List[str | None]]: Paths of the images and
                masks.
        """
        image_paths = glob.glob(
            os.path.join(dataset_path, self.TAMP_DIR, f"*{self.IMAGE_EXTENSION}")
        )
        mask_paths: List[Optional[str]] = [
            os.path.join(self.dataset_path, self._get_mask_path(image_path))
            for image_path in image_paths
        ]

        if not tampered_only:
            pris_paths = glob.glob(
                os.path.join(dataset_path, self.AUTH_DIR, f"*{self.IMAGE_EXTENSION}")
            )

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
        image_filename = image_path.split("/")[-1]
        image_name_list = ".".join(image_filename.split(".")[:-1]).split("_")
        mask_name = "_".join(image_name_list + ["edgemask"])
        mask_filename = mask_name + self.MASK_EXTENSION
        return os.path.join(self.MASKS_DIR, mask_filename)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        """
        Binarize the mask.

        Args:
            mask_image (Tensor): Mask image.

        Returns:
            Tensor: Binarized mask image.
        """
        return mask_image[self.TAMPERED_COLOR_INDEX, :, :] > 0
