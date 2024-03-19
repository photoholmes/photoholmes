from typing import Any, Dict

from torch import Tensor

from photoholmes.preprocessing.base import BasePreprocessing
from photoholmes.preprocessing.image import GetImageSize, ZeroOneRange
from photoholmes.preprocessing.pipeline import PreProcessingPipeline


class AdaptiveCFANetPreprocessing(BasePreprocessing):
    """
    Preprocessing transformation class for AdaptiveCFANet, intended to prepare
    images for processing by ensuring their dimensions are even.

    This class does not take any initialization parameters.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        image: Tensor,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Adjusts the input image dimensions to ensure they are even, as required
        by the AdaptiveCFANet architecture. This is done by trimming the image
        to the largest even dimensions that are smaller than or equal to the
        original dimensions.

        Args:
            image (Tensor): The input image tensor with shape (C, Y, X), where
                C is the number of channels, Y is the height, and X is the width.
            **kwargs (Dict[str, Any]): Additional keyword arguments that might
                be passed to the preprocessing function and need to be preserved in the
                output.

        Returns:
            Dict[str, Any]: A dictionary containing the preprocessed image tensor
                under the key 'image', along with any other keyword arguments passed
                into the method.
        """

        C, Y_o, X_o = image.shape
        image = image[:C, : Y_o - Y_o % 2, : X_o - X_o % 2]

        return {"image": image, **kwargs}


adaptive_cfa_net_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image", "image_size"],
    transforms=[
        GetImageSize(),
        ZeroOneRange(),
        AdaptiveCFANetPreprocessing(),
    ],
)
