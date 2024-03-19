from typing import List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from photoholmes.methods.base import BaseMethod, BenchmarkOutput
from photoholmes.methods.dq.utils import ZIGZAG, fft_period, histogram_period
from photoholmes.postprocessing.resizing import (
    resize_heatmap_with_trim_and_pad,
    simple_upscale_heatmap,
)


class DQ(BaseMethod):
    """
    Implementation of DQ [Lin, et.al. 2009]. The method detects forgeries
    exploiting double quantization discrepancies. It uses the DCT coefficients
    of the image to calculate the BPPM (Block Posterior Probability Map) heatmap.
    """

    def __init__(self, number_frecs: int = 8, alpha: float = 1.0, **kwargs) -> None:
        """
        Initializes the DQ class with specified parameters.

        Args:
            number_frecs (int): Number of frequency components to consider.
                Defaults to 8.
            alpha (float): Alpha value used in period detection.
                Defaults to 1.0.
        """
        super().__init__(**kwargs)
        self.number_frecs = number_frecs
        self.alpha = alpha

    def predict(
        self, dct_coefficients: NDArray, image_size: Tuple[int, int]
    ) -> NDArray:
        """
        Predicts the BPPM (Block Posterior Probability Map) heatmap from DCT
        coefficients.

        Args:
            dct_coefficients (np.ndarray): Array containing DCT coefficients of the
                image.
            image_size (Tuple[int, int]): Tuple representing the dimensions of the
                image.

        Returns:
            Tensor: The predicted BPPM heatmap as a PyTorch tensor.
        """
        M, N = dct_coefficients.shape[-2:]
        BPPM = np.zeros((M // 8, N // 8))
        for channel in range(dct_coefficients.shape[0]):
            BPPM += self._calculate_BPPM_channel(
                dct_coefficients[channel], ZIGZAG[: self.number_frecs]
            )
        BPPM_norm = torch.from_numpy(BPPM / len(dct_coefficients))
        BPPM_upsampled = simple_upscale_heatmap(BPPM_norm, 8)
        heatmap = resize_heatmap_with_trim_and_pad(BPPM_upsampled, image_size)
        return heatmap.numpy()

    def benchmark(
        self, dct_coefficients: NDArray, image_size: Tuple[int, int]
    ) -> BenchmarkOutput:
        """
        Benchmarks the DQ method using provided DCT coefficients and image size.

        Args:
            dct_coefficients (np.ndarray): DCT coefficients of the image.
            image_size (Tuple[int, int]): Dimensions of the image.

        Returns:
            BenchmarkOutput: Contains the heatmap and placeholders for mask and
                detection.
        """
        heatmap = self.predict(dct_coefficients, image_size)
        return {"heatmap": torch.from_numpy(heatmap), "mask": None, "detection": None}

    def _detect_period(self, histogram: NDArray) -> int:
        """
        Detects the repeating period in a histogram.

        Args:
            histogram (np.ndarray): The input histogram from image frequencies.

        Returns:
            int: The detected period within the histogram.
        """
        if len(histogram) < 2:
            return 1
        else:
            p_H = histogram_period(histogram, self.alpha)
            p_fft = fft_period(histogram)
            p = min(p_H, p_fft)
            return p

    def _calculate_Pu(
        self, coefficients_f: NDArray, histogram: NDArray, period: int
    ) -> NDArray:
        """
        Calculates Pu values for given frequency coefficients based on histogram.

        Args:
            coefficients_f (np.ndarray): Frequency-specific DCT coefficients.
            histogram (np.ndarray): Histogram of values for the specific frequency.
            period (int): Detected period for the histogram of the frequency.

        Returns:
            np.ndarray: Array of Pu values for the given frequency coefficients.
        """
        coefficients_f -= np.min(coefficients_f)
        M, N = coefficients_f.shape

        histogram_padded = np.pad(histogram, (0, period))
        coefficient_indices = coefficients_f.ravel()
        histogram_range = histogram_padded[
            coefficient_indices[:, np.newaxis] + np.arange(period) - 1
        ]

        Pu_f = histogram_range[:, 1] / np.sum(histogram_range, axis=1)
        Pu_f = Pu_f.reshape((M, N))

        return Pu_f

    def _calculate_BPPM_f(self, DCT_coefficients_f: NDArray) -> NDArray:
        """
        Calculates BPPM values for given DCT coefficients at a specific frequency.

        Args:
            DCT_coefficients_f (np.ndarray): DCT coefficients for a specific frequency.

        Returns:
            np.ndarray: Calculated BPPM values for the given frequency.
        """
        hmax = np.max(DCT_coefficients_f)
        hmin = np.min(DCT_coefficients_f)
        if hmax - hmin:
            hist, _ = np.histogram(
                DCT_coefficients_f, bins=hmax - hmin, range=(hmin, hmax)
            )
            p = self._detect_period(hist[1:-1])
            if p != 1:
                Pu = self._calculate_Pu(DCT_coefficients_f, hist, p)
                Pt = 1 / p
                BPPM_f = Pt / (Pu + Pt)
                saturated = (DCT_coefficients_f == DCT_coefficients_f.min()) | (
                    DCT_coefficients_f == DCT_coefficients_f.max()
                )
                BPPM_f[saturated] = 0

                return BPPM_f
        return np.zeros_like(DCT_coefficients_f)

    def _calculate_BPPM_channel(
        self, DCT_coefs: NDArray, fs: List[Tuple[int, int]]
    ) -> NDArray:
        """
        Calculates BPPM values for all frequencies within a single image channel.

        Args:
            DCT_coefs (np.ndarray): DCT coefficients for the image channel.
            fs (List[Tuple[int, int]]): List of frequency tuples to consider.

        Returns:
            np.ndarray: Aggregated BPPM values for the image channel.
        """
        M, N = DCT_coefs.shape
        BPPM = np.zeros((len(fs), M // 8, N // 8))
        for i in range(len(fs)):
            DCT_coefficients_f = DCT_coefs[fs[i][0] :: 8, fs[i][1] :: 8]
            BPPM[i] = self._calculate_BPPM_f(DCT_coefficients_f)

        return BPPM.sum(axis=0) / self.number_frecs
