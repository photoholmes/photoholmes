# Derived from code provided by Marina Gardella, please refer to
# https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000341 for an online demo

from typing import List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from skimage.util import view_as_windows

from photoholmes.methods.base import BaseMethod, BenchmarkOutput

from .utils import (
    all_image_means,
    bin_block_list,
    bin_is_valid,
    compute_low_freq_var,
    compute_output,
    compute_valid_blocks_indices,
    select_blocks_VL,
    sort_blocks_means,
    std_blocks,
    update_samples_per_bin,
)


class Noisesniffer(BaseMethod):
    """
    Noisesniffer [Gardella et al, 2021] implementation.
    The algorithm operates by analyzing the noise levels across different
    patches of the image to identify inconsistencies.
    """

    def __init__(
        self,
        w: int = 3,
        b: int = 20000,
        n: float = 0.1,
        m: float = 0.5,
        W: int = 100,
        **kwargs,
    ):
        """
        Args:
            w (int): Block size for noise analysis (default: 3).
            b (int): Number of blocks per bin for statistical analysis (default: 20000).
            n (float): Percentile for selecting blocks with the lowest energy in low
                    frequencies (default: 0.1).
            m (float): Percentile for selecting blocks with the lowest standard
                    deviation (default: 0.5).
            W (int): Cell size for NFA (Number of False Alarms) computation for region
                    growing (default: 100).
        """
        super().__init__(**kwargs)
        self.w = w
        self.b = b
        self.n = n
        self.m = m
        self.W = W

    def do_one_channel(
        self,
        ch: int,
        n: float,
        m: float,
        image: NDArray,
        w: int,
        img_means: NDArray,
        valid_blocks_indices: NDArray,
        b: int,
    ) -> Tuple[List, List]:
        """
        Run Noisesniffer in one channel of input image.

        Args:
            ch (int): Index of the channel to be processed.
            n (float): Percentile of blocks with the lowest energy in low frequencies.
            m (float): Percentile of blocks with the lowest standard deviation.
            image (NDArray): Input image array.
            w (int): Block size.
            img_means (NDArray): Means of all blocks in the image.
            valid_blocks_indices (NDArray): Indices of blocks considered valid.
            b (int): Number of blocks per bin.

        Returns:
            List: List of valid blocks in the current channel.
            List: List of selected blocks.
        """
        V = []
        S = []

        #  extract image blocks in channel ch
        img_blocks = view_as_windows(image[:, :, ch], w).reshape(-1, w, w)
        # compute low freq variance of the blocks
        low_freq_var = compute_low_freq_var(img_blocks, w).reshape(-1)
        # sort valid blocks according to their mean
        blocks_sorted_means = sort_blocks_means(ch, img_means, valid_blocks_indices)
        # update the number of samples per bin
        b = update_samples_per_bin(b, len(valid_blocks_indices))
        # compute number of bins
        num_bins = int(round(len(valid_blocks_indices) / b))
        for Bin in range(num_bins):
            # list the blocks in the bin
            blocks_in_bin = bin_block_list(num_bins, Bin, blocks_sorted_means, b)

            # select blocks according to their variance in low freqs
            blocks_select_LF_var = select_blocks_VL(b, n, low_freq_var, blocks_in_bin)

            # sort selected blocks according to their std
            blocks_stds_sorted = std_blocks(img_blocks, blocks_select_LF_var)

            if bin_is_valid(b, n, m, blocks_stds_sorted):
                for k, pos in enumerate(blocks_stds_sorted):
                    V.append(pos)
                    if k < int(b * n * m):
                        S.append(pos)
        return V, S

    def predict(  # type: ignore[override]
        self, image: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Run Noisesniffer on an image.

        Args:
            image (NDArray): Input image.

        Returns:
            NDArray: Mask of detected forgeries.
            NDarray: Image with lowest standard deviation blocks painted
                red.
        """
        image = image.astype(float)
        valid_blocks_indices = compute_valid_blocks_indices(image, self.w)
        img_means = all_image_means(image, self.w)

        V: List = []
        S: List = []

        for ch in range(image.shape[2]):
            V_ch, S_ch = self.do_one_channel(
                ch,
                self.n,
                self.m,
                image,
                self.w,
                img_means,
                valid_blocks_indices,
                self.b,
            )
            V = np.concatenate((V, V_ch))
            S = np.concatenate((S, S_ch))

        mask, img_paint = compute_output(image, self.w, self.W, self.m, V, S)
        return mask, img_paint

    def benchmark(self, image: NDArray) -> BenchmarkOutput:  # type: ignore[override]
        """
        Benchmarks the Noisesniffer method using the provided image.

        Args:
            image (Tensor): Input image tensor.

        Returns:
            BenchmarkOutput: Contains the mask and detection and placeholder for
                heatmap.
        """
        mask, _ = self.predict(image)

        detection = float(np.any(mask))
        detection_tensor = torch.tensor(detection).unsqueeze(0)

        mask_tensor = torch.from_numpy(mask)

        return BenchmarkOutput(
            heatmap=None,
            mask=mask_tensor,
            detection=detection_tensor,
        )
