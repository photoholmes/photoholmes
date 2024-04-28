import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from photoholmes.preprocessing import PreProcessingPipeline
from photoholmes.utils.image import read_image, read_jpeg_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AttributeOverrideError(NotImplementedError):
    """
    Exception raised when a subclass fails to override a required class attribute.
    """

    def __init__(self, attribute_name: str):
        message = f"Subclasses must override {attribute_name}"
        super().__init__(message)


class BaseDataset(ABC, Dataset):
    """
    Base class for datasets.

    Subclasses must override the IMAGE_EXTENSION and MASK_EXTENSION attributes.
    The _get_paths and _get_mask_path methods must be implemented as well, and in some
    cases binarize_mask must alos be overriden.
    """

    IMAGE_EXTENSION: Union[str, List[str]]
    MASK_EXTENSION: Union[str, List[str]]

    def __init__(
        self,
        dataset_path: str,
        preprocessing_pipeline: Optional[PreProcessingPipeline] = None,
        load: List[
            Literal[
                "image",
                "dct_coefficients",
                "qtables",
            ]
        ] = [
            "image",
            "dct_coefficients",
            "qtables",
        ],
        tampered_only: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (str): Path to the dataset.
            preprocessing_pipeline (Optional[PreProcessingPipeline]): Preprocessing
                pipeline to apply to the images.
            load (List[Literal["image", "dct_coefficients", "qtables"]]): List of
                items to load. Possible values are "image", "dct_coefficients" and
                "qtables". If the preprocessing_pipeline is not None, the load
                attribute will be ignored and the preprocessing pipeline inputs will
                be used instead.
            tampered_only (bool): If True, only load tampered images.

        Raises:
            FileNotFoundError: If the dataset_path does not exist.
            AttributeOverrideError: If the subclass has not overridden the
                IMAGE_EXTENSION and MASK_EXTENSION attributes.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Directory {dataset_path} does not exist.")

        self.load = preprocessing_pipeline.inputs if preprocessing_pipeline else load
        self.load_jpeg_data = "dct_coefficients" in self.load or "qtables" in self.load
        self.load_image_data = "image" in self.load

        if self.load_jpeg_data:
            self.check_jpeg_warning()
        self.check_attribute_override()

        self.dataset_path = dataset_path
        self.tampered_only = tampered_only

        self.preprocessing_pipeline = preprocessing_pipeline

        if preprocessing_pipeline:
            if set(load) != set(preprocessing_pipeline.inputs):
                logger.warning(
                    "The load attribute and the preprocessing pipeline inputs do not "
                    f"match. Using the preprocessing pipeline inputs: "
                    f"{preprocessing_pipeline.inputs}"
                )

        self.image_paths, self.mask_paths = self._get_paths(dataset_path, tampered_only)

    @abstractmethod
    def _get_paths(
        self, dataset_path, tampered_only
    ) -> Tuple[List[str], List[str] | List[str | None]]:
        """
        Abstract method that returns an ordered list of image and mask paths, mapped
        in the correct order.
        The correct implementation in a child class must follow:
         - Make use of the dataset_path and tampered_only arguments.
         - In the case of pristine images, the corresponding mask path must be set to 'None'.
         - Mask paths must be obtained by a correspondance of the image path,
         using the _get_mask_path method.

        Args:
            dataset_path (str): Path to the dataset.
            tampered_only (bool): Whether to load only the tampered images.

        Returns:
            Tuple[List[str], List[str] | List[str | None]]: Tuple with lists of image and mask paths (or None in pristine images).
        """
        pass

    @abstractmethod
    def _get_mask_path(self, image_path: str) -> str:
        """Abstract method that returns the corresponding mask path for a given image path."""
        pass

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Dict, Tensor, str]:
        """Return the item at the given index."""
        x, mask, image_name = self._get_data(idx)
        if self.preprocessing_pipeline is not None:
            x = self.preprocessing_pipeline(**x)
        return x, mask, image_name

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
        image_name = image_path.split("/")[-1].split(".")[0]

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

    def _binarize_mask(self, mask_image: Tensor) -> Tensor:
        """
        Overridable method to binarize the mask image. Binarized masks are boolean
        tensors of one channel, regarding any degree of tampering as True.
        Arguments:
            mask_image (Tensor): Original mask image.
        Outputs:
            Tensor: Binarized mask image.
        """
        assert (mask_image <= 1).all()
        return (mask_image == 1).float()

    def check_attribute_override(self):
        """
        Check that the subclass has overridden IMAGE_EXTENSION and MASK_EXTENSION.
        Raises an error if not.
        """
        if not hasattr(type(self), "IMAGE_EXTENSION"):
            raise AttributeOverrideError("IMAGE_EXTENSION")
        if not hasattr(type(self), "MASK_EXTENSION"):
            raise AttributeOverrideError("MASK_EXTENSION")

    def check_jpeg_warning(self):
        """
        Check if the images are in JPEG format. If not, a warning is issued.
        """
        if not isinstance(self.IMAGE_EXTENSION, list):
            image_ext = [self.IMAGE_EXTENSION]
        else:
            image_ext = self.IMAGE_EXTENSION
        if not all(
            [
                ext in [".jpg", ".jpeg", ".JPG", ".JPEG", "jpg", "jpeg", "JPEG", "JPG"]
                for ext in image_ext
            ]
        ):
            logger.warning(
                "Not all images are in JPEG format. When needed, an approximation will "
                "be loaded by compressing the image in quality 100."
            )
