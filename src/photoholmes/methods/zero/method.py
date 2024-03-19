from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.fftpack import dctn
from torch import any as torch_any
from torch import from_numpy

from photoholmes.methods.base import BaseMethod, BenchmarkOutput

from .utils import log_nfa


class Zero(BaseMethod):
    """
    Implementation of the Zero method [Nikoukhah et al., 2021].

    The method is based on the detection of JPEG compression and grid
    alignment abnormalities. It is also capable of detecting local image
    forgeries such as copy-move.

    The original implementation(languages: C, Python) is available at:
    https://github.com/tinankh/ZERO
    """

    def __init__(self, no_vote: int = -1, **kwargs) -> None:
        """
        Args:
            no_vote (int): Value to be used as no vote. Default is -1.
            kwargs: Additional arguments to be passed to the BaseMethod class.
        """
        self.no_vote = no_vote
        super().__init__(**kwargs)

    def predict(self, image: NDArray) -> Tuple[NDArray, NDArray, int]:
        """
        Run Zero on a image. The image is expected to be in YCbCr format. The
        methods is run over the luminance channel.

        Args:
            image (np.ndarray): Input image.

        Returns:
            Tuple[NDArray, NDArray, int]: Forgery mask, votes and main grid.
        """
        luminance = image[..., 0]
        votes = self.compute_grid_votes_per_pixel(luminance)
        main_grid = self.detect_global_grids(votes)
        forgery_mask = self.detect_forgeries(votes, main_grid)

        return forgery_mask, votes, main_grid

    def benchmark(self, image: NDArray) -> BenchmarkOutput:
        """
        Benchmarks the Zero method using the provided image.

        Args:
            image (Tensor): Input image tensor.

        Returns:
            BenchmarkOutput: Contains the mask and detection and placeholder for
            heatmap.
        """
        forgery_mask, _, _ = self.predict(image)
        mask = from_numpy(forgery_mask)
        detection = torch_any(mask).float().unsqueeze(0)
        return {
            "heatmap": None,
            "mask": mask,
            "detection": detection,
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
        votes = np.zeros_like(luminance, dtype=np.int32)

        for x in range(X - 7):
            for y in range(Y - 7):
                block = luminance[y : y + 8, x : x + 8]
                const_along_x = np.all(block[:, :] == block[:1, :])
                const_along_y = np.all(block[:, :] == block[:, :1])

                dct = dctn(block, type=2, norm="ortho")
                z = (np.abs(dct) < 0.5).sum()

                mask_zeros = z == zeros[y : y + 8, x : x + 8]
                mask_greater = z > zeros[y : y + 8, x : x + 8]

                votes[y : y + 8, x : x + 8][mask_zeros] = self.no_vote
                zeros[y : y + 8, x : x + 8][mask_greater] = z
                votes[y : y + 8, x : x + 8][mask_greater] = (
                    self.no_vote
                    if const_along_x or const_along_y
                    else (x % 8) + (y % 8) * 8
                )
        votes[:7, :] = votes[-7:, :] = votes[:, :7] = votes[:, -7:] = self.no_vote

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
        max_votes = 0
        most_voted_grid = self.no_vote
        p = 1.0 / 64.0
        for x in range(X):
            for y in range(Y):
                if votes[y, x] >= 0 and votes[y, x] < 64:
                    grid = votes[y, x]
                    grid_votes[grid] += 1
                    if grid_votes[grid] > max_votes:
                        max_votes = grid_votes[grid]
                        most_voted_grid = grid
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

        return most_voted_grid if grid_meaningful else self.no_vote

    def detect_forgeries(self, votes: NDArray, grid_to_exclude: int) -> NDArray:
        """
        Detects forgery mask from a grid votes map and a grid index to exclude.

        Args:
            votes (NDArray): Grid votes per pixel. Each pixel votes 1 of the 64
                possible grids.
            grid_to_exclude (int): Grid index to exclude.

        Returns:
            NDArray: forgery mask.
        """
        W = 9
        grid_max = 63
        p = 1.0 / 64.0
        Y, X = votes.shape
        N_tests = (64 * X * Y) ** 2

        used = np.full_like(votes, False)
        reg_x = np.zeros(votes.shape[0] * votes.shape[1], dtype=int)
        reg_y = np.zeros(votes.shape[0] * votes.shape[1], dtype=int)
        forgery_mask = np.zeros_like(votes)

        min_size = np.ceil(64.0 * np.log10(N_tests) / np.log10(64.0)).astype(int)

        for x in range(X):
            for y in range(Y):
                if (
                    not used[y, x]
                    and votes[y, x] != grid_to_exclude
                    and votes[y, x] >= 0
                    and votes[y, x] <= grid_max
                ):
                    grid = votes[y, x]
                    corner_0 = corner_1 = np.array([x, y])
                    used[y, x] = True
                    reg_x[0] = x
                    reg_y[0] = y
                    reg_size = 1
                    i = 0

                    while i < reg_size:
                        lower_xx = max(reg_x[i] - W, 0)
                        higher_xx = min(reg_x[i] + W + 1, X)
                        lower_yy = max(reg_y[i] - W, 0)
                        higher_yy = min(reg_y[i] + W + 1, Y)
                        for xx in range(lower_xx, higher_xx):
                            for yy in range(lower_yy, higher_yy):
                                if not used[yy, xx] and votes[yy, xx] == grid:
                                    used[yy, xx] = True
                                    reg_x[reg_size] = xx
                                    reg_y[reg_size] = yy
                                    reg_size += 1

                                    corner_0 = np.min(
                                        np.vstack([corner_0, np.array([xx, yy])]),
                                        axis=1,
                                    )
                                    corner_1 = np.max(
                                        np.vstack([corner_0, np.array([xx, yy])]),
                                        axis=1,
                                    )
                        i += 1
                    if reg_size >= min_size:
                        n = int(
                            (corner_1[0] - corner_0[0] + 1)
                            * (corner_1[1] - corner_1[1] + 1)
                            // 64
                        )
                        k = int(reg_size // 64)
                        lnfa = log_nfa(N_tests, np.array([k]), n, p)[0]
                        if lnfa < 0.0:
                            idxs = np.array([reg_x[:reg_size], reg_y[:reg_size]]).T
                            forgery_mask[idxs[:, 1], idxs[:, 0]] = 1

        return forgery_mask.astype(float)
