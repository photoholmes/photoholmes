import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

from .base import BaseDataset


class CasiaBaseDataset(BaseDataset):
    """
    Base class for the CASIA 1.0 dataset.

    Directory structure:
    img_dir (CASIA 1.0 dataset)
    ├── Au
    │   └── [Authentic images in jpg]
    ├── Tp
    │   ├── CM
    │   │   └── [Copy Move images in jpg]
    |   ├── Sp
    |       └── [Spliced images in jpg]
    ├── CASIA 1.0 groundtruth
    │   ├── CM
    │   │   └── [Copy Move masks in png]
    |   ├── Sp
    |       └── [Spliced masks in png]
    └── Possibly more files

    Subclasses should define the IMAGES_SUB_DATASET_DIR and MASK_SUB_DATASET_DIR
    """

    IMAGES_TAMPERED_DIR: str
    MASK_TAMPERED_DIR: str
    AUTH_DIR: str = "Au"
    IMAGE_EXTENSION: str = ".jpg"
    MASK_EXTENSION: str = ".png"

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
            os.path.join(
                dataset_path, self.IMAGES_TAMPERED_DIR, f"*{self.IMAGE_EXTENSION}"
            )
        )
        mask_paths: List[Optional[str]] = [
            os.path.join(dataset_path, self._get_mask_path(image_path))
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
        mask_name = "_".join(image_name_list + ["gt"])
        mask_filename = mask_name + self.MASK_EXTENSION
        return os.path.join(self.MASK_TAMPERED_DIR, mask_filename)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        """
        Binarize the mask.

        Args:
            mask_image (Tensor): Mask image.

        Returns:
            Tensor: Binarized mask image.
        """
        return mask_image[0, :, :] > 0


class Casia1SplicingDataset(CasiaBaseDataset):
    """
    Class for the CASIA 1.0 Splicing (Sp) subset.
    """

    IMAGES_TAMPERED_DIR = "Tp/Sp"
    MASK_TAMPERED_DIR = "CASIA 1.0 groundtruth/Sp"


class Casia1CopyMoveDataset(CasiaBaseDataset):
    """
    Class for the CASIA 1.0 Copy Move (CM) subset.
    """

    IMAGES_TAMPERED_DIR = "Tp/CM"
    MASK_TAMPERED_DIR = "CASIA 1.0 groundtruth/CM"
