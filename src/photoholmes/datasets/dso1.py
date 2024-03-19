import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .base import BaseDataset


class DSO1Dataset(BaseDataset):
    """
    Directory structure:
    img_dir (tifs-database)
    ├── DSO-1
    │   ├── [images in png, normal for untampered, splicing for forged]
    ├── DSO-1-Fake-Images-Masks
    │   └── [masks in png]
    └── Possibly more folders
    """

    IMAGE_DIR = "DSO-1"
    MASKS_DIR = "DSO-1-Fake-Images-Masks"
    IMAGE_EXTENSION = ".png"
    MASK_EXTENSION = ".png"
    TAMPERED_TAG = "splicing"

    def _get_paths(
        self, dataset_path: str, tampered_only: bool
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        """
        Get the paths of the images and masks in the dataset.

        Args:
            dataset_path (str): Path to the dataset.
            only_load_tampered (bool): Whether to load only the tampered images.

        Returns:
            Tuple[List[str], List[str] | List[str | None]]: Paths of the images and
                masks.
        """
        tag = self.TAMPERED_TAG if tampered_only else ""
        image_paths = glob.glob(
            os.path.join(dataset_path, self.IMAGE_DIR, f"{tag}*{self.IMAGE_EXTENSION}")
        )
        mask_paths: List[Optional[str]] = [
            self._get_mask_path(image_path) if self._is_tampered(image_path) else None
            for image_path in image_paths
        ]
        return image_paths, mask_paths

    def _is_tampered(self, image_path: str) -> bool:
        """
        Check if the image is tampered.

        Args:
            image_path (str): Path to the image.

        Returns:
            bool: True if the image is tampered, False otherwise.
        """
        filename = os.path.basename(image_path)
        tag = filename.split("-")[0]
        return tag == self.TAMPERED_TAG

    def _get_mask_path(self, image_path: str) -> str:
        """
        Get the path of the mask for the given image path.

        Args:
            image_path (str): Path to the image.

        Returns:
            str: Path to the mask.
        """
        img_dir = "/".join(image_path.split("/")[:-2])
        im_filename = image_path.split("/")[-1]
        mask_path = os.path.join(img_dir, self.MASKS_DIR, im_filename)
        return mask_path

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        """
        Binarize the mask.

        Args:
            mask_image (Tensor): Mask image.

        Returns:
            Tensor: Binarized mask image.
        """
        return mask_image[0, :, :] == 0
