import os

from torch import Tensor

from .casia1 import CasiaBaseDataset
from .columbia import ColumbiaDataset
from .dso1 import DSO1Dataset


class Casia1SplicingOSNDataset(CasiaBaseDataset):
    """
    Class for the CASIA 1.0 Splicing subset with Online Social Networks (OSN)
    modifications.

    Directory structure:
    img_dir (OSN dataset)
    ├── Casia_Facebook
    │   ├── CM
    │   │   └── [Copy Move images in jpg]
    │   ├── Sp
    │   │   └── [Spliced images in jpg]
    │   ├── CASIA_GT
    │       └── [Copy Move and Spliced masks in png]
    └── Possibly more files
    """

    IMAGES_TAMPERED_DIR: str = "Casia_Facebook/Sp"
    MASK_TAMPERED_DIR: str = "Casia_Facebook/CASIA_GT"
    tampered_only: bool = True


class Casia1CopyMoveOSNDataset(CasiaBaseDataset):
    """
    Class for the CASIA 1.0 Copy Move subset with Online Social Networks (OSN)
    modifications.

    Directory structure:
    img_dir (OSN dataset)
    ├── Casia_Facebook
    │   ├── CM
    │   │   └── [Copy Move images in jpg]
    │   ├── Sp
    │   │   └── [Spliced images in jpg]
    │   └── CASIA_GT
    │       └── [Copy Move and Spliced masks in png]
    ├── Columbia_Facebook
    └── DSO_Facebook
    """

    IMAGES_TAMPERED_DIR: str = "Casia_Facebook/CM"
    MASK_TAMPERED_DIR: str = "Casia_Facebook/CASIA_GT"
    tampered_only: bool = True


class ColumbiaOSNDataset(ColumbiaDataset):
    """
    Class for the Columbia Uncompressed Image Splicing Detection dataset with
    Online Social Networks (OSN) modifications.

    Directory structure:
    img_dir (OSN dataset)
    ├── Casia_Facebook
    ├── Columbia_Facebook
    │   ├── Columbia_GT
    │   │   └── [masks in PNG]
    │   └── [images in JPG]
    └── DSO_Facebook
    """

    TAMP_DIR: str = "Columbia_Facebook"
    IMAGE_EXTENSION: str = ".jpg"
    MASKS_DIR: str = "Columbia_Facebook/Columbia_GT"
    MASK_EXTENSION: str = ".png"
    tampered_only: bool = True

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
        return os.path.join(self.MASKS_DIR, mask_filename)


class DSO1OSNDataset(DSO1Dataset):
    """
    Class for the DSO-1 dataset with Online Social Networks (OSN) modifications.

    Directory structure:
    img_dir (OSN dataset)
    ├── Casia_Facebook
    ├── Columbia_Facebook
    └── DSO_Facebook
        ├── [Images in jpg]
        └── DSO_GT
            └── [masks in png]
    """

    TAMP_DIR: str = "DSO_Facebook"
    IMAGE_EXTENSION: str = ".jpg"
    IMAGE_DIR: str = "DSO_Facebook"
    MASK_EXTENSION: str = "_gt.png"
    MASKS_DIR: str = "DSO_Facebook/DSO_GT"
    tampered_only: bool = True

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
        im_filename = im_filename.replace(self.IMAGE_EXTENSION, self.MASK_EXTENSION)
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
        return mask_image[0, :, :] > 0
