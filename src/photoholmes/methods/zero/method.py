import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.fftpack import dctn
from torch import any as torch_any
from torch import from_numpy as torch_from_numpy

from photoholmes.methods.base import BaseMethod, BenchmarkOutput

from .utils import closing, log_nfa

logger = logging.getLogger(__name__)


class Zero(BaseMethod):
    """
    Implementation of the Zero method [Nikoukhah et al., 2021].

    The method is based on the detection of JPEG compression and grid
    alignment abnormalities. It is also capable of detecting local image
    forgeries such as copy-move.

    The original implementation(languages: C, Python) is available at:
    https://github.com/tinankh/ZERO
    """

    NO_VOTE: int = -1

    def __init__(self, missing_grids: bool = False) -> None:
        """
        Args:
            missing_grids (bool): Whether to detect missing grids or not.
        """
        super().__init__()

        self.missing_grids = missing_grids

    def predict(  # type: ignore[override]
        self, image: NDArray, image_99: Optional[NDArray] = None
    ) -> Tuple[NDArray, NDArray, NDArray, Optional[NDArray]]:
        """
        Run Zero on a image. The method is run over the iluminance of the image,
        so it expects a grasycale image.

        The method's first output is the joint_mask, which is the union of the forgery mask and
        the missing grid mask if `missing_grids` is set to True

        If `missing_grids` is set to false, the joint_mask is the same as the forgery_mask.

        Args:
            image (np.ndarray): iluminance of input image.
            image_99 (np.ndarray): iluminance of input image with jpeg compression quality 99.
                (see zero/preprocessing.py to see how to generate it.)

        Returns:
            Tuple[NDArray, NDArray, Optional[NDArray]]: joint mask, forgery_mask, votes and
                missing_grid if selected, else None.
        """
        luminance = image[..., 0]
        votes = self.compute_grid_votes_per_pixel(luminance)
        main_grid = self.detect_global_grids(votes)
        forgery_mask = self.detect_forgeries(votes, main_grid, 63)

        if self.missing_grids:
            if image_99 is None:
                logger.warning(
                    "`missing_grids` is set, but `image_99` is not provided. "
                    "Skipping missing grids mask calculation."
                )
            if main_grid > -1 and image_99 is not None:
                votes_jpeg = self.compute_grid_votes_per_pixel(image_99[..., 0])

                # do not count votes from the main grid
                votes_jpeg[votes == main_grid] = self.NO_VOTE

                mask_missing_grids = self.detect_forgeries(
                    votes_jpeg, grid_to_exclude=self.NO_VOTE, grid_max=0
                )
            else:
                mask_missing_grids = np.zeros_like(forgery_mask)

            joint_mask = np.logical_or(forgery_mask, mask_missing_grids)

        else:
            mask_missing_grids = None
            joint_mask = forgery_mask.copy()

        return joint_mask, forgery_mask, votes, mask_missing_grids

    def benchmark(self, image: NDArray, image_99: NDArray) -> BenchmarkOutput:
        """
        Benchmarks the Zero method using the provided image.

        Args:
            image (Tensor): Input image tensor.
            image_99 (Tensor): Input image tensor with JPEG compression quality 99.

        Returns:
            BenchmarkOutput: Contains the mask and detection and placeholder for
            heatmap.
        """
        forgery_mask, *_ = self.predict(image, image_99)
        mask = torch_from_numpy(forgery_mask).float()
        detection = torch_any(mask).float().unsqueeze(0)

        return {
            "mask": mask,
            "detection": detection,
            "heatmap": None,
        }

    def compute_grid_votes_per_pixel(self, luminance: NDArray) -> NDArray:
        """
        Compute the grid votes per pixel.

        Args:
            luminance (NDArray): Input luminance channel.

        Returns:
            NDArray: Grid votes per pixel.
        """
        Y, X = luminance.shape
        zeros = np.zeros_like(luminance, dtype=np.int32)
        votes = np.full_like(luminance, self.NO_VOTE, dtype=np.int32)

        for x in range(X - 7):
            for y in range(Y - 7):
                block = luminance[y : y + 8, x : x + 8]
                const_along_x = np.all(block[:, :] == block[:1, :])
                const_along_y = np.all(block[:, :] == block[:, :1])

                dct = dctn(block, type=2, norm="ortho")
                dct[0, 0] = 1  # Discard DC component
                z = (np.abs(dct) < 0.5).sum()

                mask_zeros = z == zeros[y : y + 8, x : x + 8]
                mask_greater = z > zeros[y : y + 8, x : x + 8]

                votes[y : y + 8, x : x + 8][mask_zeros] = self.NO_VOTE
                zeros[y : y + 8, x : x + 8][mask_greater] = z
                votes[y : y + 8, x : x + 8][mask_greater] = (
                    self.NO_VOTE
                    if const_along_x or const_along_y
                    else (x % 8) + (y % 8) * 8
                )

        votes[:7, :] = votes[-7:, :] = votes[:, :7] = votes[:, -7:] = self.NO_VOTE

        return votes

    def detect_global_grids(self, votes: NDArray) -> int:
        """
        Detects the main estimated grid.

        Args:
            votes (NDArray): Grid votes per pixel. Each pixel votes 1 of the 64
                possible grids.

        Returns:
            int: Main detected grid.
        """
        Y, X = votes.shape
        grid_votes = np.zeros(64)
        most_voted_grid = self.NO_VOTE
        p = 1.0 / 64.0

        valid_votes = np.argwhere((votes >= 0) * (votes < 64))
        grid_votes, _ = np.histogram(
            votes[valid_votes[:, 0], valid_votes[:, 1]], bins=np.arange(65)
        )
        most_voted_grid = int(np.argmax(grid_votes))

        N_tests = (64 * X * Y) ** 2
        ks = np.floor(grid_votes / 64) - 1
        n = np.ceil(X * Y / 64)
        p = 1 / 64
        lnfa_grids = log_nfa(N_tests, ks, n, p)

        grid_meaningful = (
            most_voted_grid >= 0
            and most_voted_grid < 64
            and lnfa_grids[most_voted_grid] < 0.0
        )

        return most_voted_grid if grid_meaningful else self.NO_VOTE

    def detect_forgeries(
        self, votes: NDArray, grid_to_exclude: int, grid_max: int
    ) -> NDArray:
        """
        Detects forgery mask from a grid votes map and a grid index to exclude.

        Args:
            votes (NDArray): Grid votes per pixel. Each pixel votes 1 of the 64
                possible grids.
            grid_to_exclude (int): Grid index to exclude.
            grid_max (int): Maximum grid index.

        Returns:
            NDArray: forgery mask.
        """
        W = 9
        p = 1.0 / 64.0
        Y, X = votes.shape
        N_tests = (64 * X * Y) ** 2

        used = np.full_like(votes, False)
        reg_x = np.zeros(votes.shape[0] * votes.shape[1], dtype=np.int16)
        reg_y = np.zeros(votes.shape[0] * votes.shape[1], dtype=np.int16)
        forgery_mask = np.zeros_like(votes)

        min_size = np.ceil(64.0 * np.log10(N_tests) / np.log10(64.0)).astype(np.int16)

        for x in range(X):
            for y in range(Y):
                if (
                    (not used[y, x])
                    and (votes[y, x] != grid_to_exclude)
                    and (votes[y, x] >= 0)
                    and (votes[y, x] <= grid_max)
                ):
                    grid = votes[y, x]
                    x0, y0 = x, y
                    x1, y1 = x, y
                    used[y, x] = True

                    reg_x[0] = x
                    reg_y[0] = y
                    reg_size = 1

                    i = 0
                    while i < reg_size:
                        low_x = max(reg_x[i] - W, 0)
                        high_x = min(reg_x[i] + W + 1, X)
                        low_y = max(reg_y[i] - W, 0)
                        high_y = min(reg_y[i] + W + 1, Y)
                        for xx in range(low_x, high_x):
                            for yy in range(low_y, high_y):
                                if not used[yy, xx] and votes[yy, xx] == grid:
                                    used[yy, xx] = True
                                    reg_x[reg_size] = xx
                                    reg_y[reg_size] = yy
                                    reg_size += 1

                                    x0 = min(x0, xx)
                                    y0 = min(y0, yy)
                                    x1 = max(x1, xx)
                                    y1 = max(y1, yy)

                        i += 1

                    if reg_size >= min_size:
                        n = (x1 - x0 + 1) * (y1 - y0 + 1) // 64
                        k = int(reg_size / 64)
                        lnfa = log_nfa(N_tests, np.array([k]), n, p)[0]

                        if lnfa < 0.0:
                            idxs = np.array([reg_x[:reg_size], reg_y[:reg_size]]).T
                            forgery_mask[idxs[:, 1], idxs[:, 0]] = 1

        forgery_mask = closing(forgery_mask, W)

        return forgery_mask
