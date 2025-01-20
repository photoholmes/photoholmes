from random import randint
from typing import Dict, Tuple, Union

import torch
from torch import Tensor

from photoholmes.preprocessing.base import BasePreprocessing
from photoholmes.preprocessing.image import GetImageSize
from photoholmes.preprocessing.pipeline import PreProcessingPipeline


def get_binary_volume(x: Tensor, T: int = 20) -> Tensor:
    x_vol = torch.zeros(size=(T + 1, x.shape[1], x.shape[2]), device=x.device)

    x_vol[0] += (x == 0).float().squeeze()

    indices = torch.arange(1, T, device=x.device).unsqueeze(1).unsqueeze(2)
    x_vol[1:T] += ((x == indices) | (x == -indices)).float().squeeze()

    x_vol[T] += (x >= T).float().squeeze()
    x_vol[T] += (x <= -T).float().squeeze()

    return x_vol


class CatnetPreprocessing(BasePreprocessing):
    """
    Preprocessing transformation class for CatNet, intended to prepare the images
    and create the DCT volumes needed as input for the net.
    """

    def __init__(self, n_dct_channels: int = 1, T: int = 20):
        """
        Initialization of Catnet preprocessing class.

        Args:
            n_dct_channels (int): Number of DCT channels to use. Defaults to 1.
            T (int): Number of quantization levels. Defaults to 20.
        """
        self.n_dct_channels = n_dct_channels
        self.T = T

    def __call__(
        self,
        image: Tensor,
        dct_coefficients: Tensor,
        qtables: Tensor,
        **kwargs,
    ) -> Dict[str, Union[Tensor, Tuple[int, int]]]:
        """
        Preprocesses the input image and DCT coefficients to prepare them for the net.
        Creates the DCT volumes needed as input for the net.

        Args:
            image (Tensor): The input image tensor with shape (C, Y, X), where
                C is the number of channels, Y is the height, and X is the width.
            dct_coefficients (Tensor): The DCT coefficients tensor with shape (C, Y, X),
                where C is the number of channels, Y is the height, and X is the width.
            qtables (Tensor): The quantization tables tensor with shape (C, 64), where
                C is the number of channels.
             **kwargs: Additional keyword arguments that might
                be passed to the preprocessing function and need to be preserved in the
                output.

        Returns:
            Dict[str, Union[Tensor, Tuple[int, int]]]: A dictionary containing the
                preprocessed image tensor with the corresponding DCT volume under the
                key 'x', the quantization tables tensor under the key 'qtable', and the
                image size as a tuple under the key 'image_size', along with any other
                keyword arguments passed into the method.
        """
        h, w = image.shape[-2:]

        crop_size = ((h // 8) * 8, (w // 8) * 8)
        if h < crop_size[0] or w < crop_size[1]:
            temp = torch.full(
                (max(h, crop_size[0]), max(w, crop_size[1]), 3),
                127.5,
                device=image.device,
            )
            temp[: image.shape[0], : image.shape[1], :] = image
            max_h = max(
                crop_size[0],
                max([dct_coefficients[c].shape[0] for c in range(self.n_dct_channels)]),
            )
            max_w = max(
                crop_size[1],
                max([dct_coefficients[c].shape[1] for c in range(self.n_dct_channels)]),
            )
            for i in range(self.n_dct_channels):
                temp = torch.full(
                    (max_h, max_w), 0.0, device=image.device
                )  # pad with 0
                temp[: dct_coefficients[i].shape[0], : dct_coefficients[i].shape[1]] = (
                    dct_coefficients[i][:, :]
                )
                dct_coefficients[i] = temp

        s_r = (randint(0, max(h - crop_size[0], 0)) // 8) * 8
        s_c = (randint(0, max(w - crop_size[1], 0)) // 8) * 8
        image = image[:, s_r : s_r + crop_size[0], s_c : s_c + crop_size[1]]
        dct_coefficients = dct_coefficients[
            : self.n_dct_channels, s_r : s_r + crop_size[0], s_c : s_c + crop_size[1]
        ]
        image = (image - 127.5) / 127.5
        t_dct_vols = get_binary_volume(dct_coefficients, T=self.T)

        x = torch.concatenate((image, t_dct_vols))

        return {"x": x, "qtable": qtables, **kwargs}


catnet_preprocessing = PreProcessingPipeline(
    inputs=["image", "qtables", "dct_coefficients"],
    outputs_keys=["x", "qtable", "image_size"],
    transforms=[GetImageSize(), CatnetPreprocessing()],
)
