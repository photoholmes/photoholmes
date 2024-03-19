import glob
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from photoholmes.utils.image import read_image, read_jpeg_data

from .base import BaseDataset


class RealisticTamperingDataset(BaseDataset):
    """
    Class for the Realistic Tampering dataset.

    Directory structure:
    img_dir (realistic-tampering-dataset)
    ├── Canon_60D
    │   ├── ground-truth
    |   |   └── [masks in PNG]
    │   ├── pristine
    |   |   └── [imgs in TIF]
    │   └── tampered-realistic
    |       └── [imgs in TIF]
    ├── Nikon_D90
    │   └── ...[idem above]
    ├── Nikon_D7000
    │   └── ...[idem above]
    ├── Sony_A57
    │   └── ...[idem above]
    └── readme.md
    """

    CAMERA_FOLDERS: List[str] = ["Canon_60D", "Nikon_D90", "Nikon_D7000", "Sony_A57"]
    TAMP_DIR: str = "tampered-realistic"
    AUTH_DIR: str = "pristine"
    MASKS_DIR: str = "ground-truth"
    IMAGE_EXTENSION: str = ".TIF"
    MASK_EXTENSION: str = ".PNG"
    TAMPERED_INTENSITY_THRESHOLD: int = 10  # 3 level masks. Gray level is tampered.

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
        image_paths = []
        mask_paths = []
        for camera_dir in self.CAMERA_FOLDERS:
            cam_im_paths, cam_mk_paths = self._get_camera_paths(
                os.path.join(dataset_path, camera_dir), tampered_only
            )
            image_paths += cam_im_paths
            mask_paths += cam_mk_paths
        return image_paths, mask_paths

    def _get_camera_paths(
        self, camera_dir, tampered_only
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
        image_paths = glob.glob(
            os.path.join(camera_dir, self.TAMP_DIR, f"*{self.IMAGE_EXTENSION}")
        )

        mask_paths: List[Optional[str]] = [
            os.path.join(self.dataset_path, self._get_mask_path(image_path))
            for image_path in image_paths
        ]

        if not tampered_only:
            pris_paths = glob.glob(
                os.path.join(camera_dir, self.AUTH_DIR, f"*{self.IMAGE_EXTENSION}")
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
        cam_dir = image_path.split("/")[-3]
        image_filename = image_path.split("/")[-1]
        name = ".".join(image_filename.split(".")[:-1])
        mask_filename = name + self.MASK_EXTENSION
        return os.path.join(cam_dir, self.MASKS_DIR, mask_filename)

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        """
        Binarize the mask.

        Args:
            mask_image (Tensor): Mask image.

        Returns:
            Tensor: Binarized mask image.
        """
        return mask_image[0, :, :] > self.TAMPERED_INTENSITY_THRESHOLD

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
