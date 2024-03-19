# Derived from code provided by Marina Gardella, please refer to
# https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000341 for an online demo
import logging
from typing import List, Tuple

import cv2
import mpmath
import numpy as np
from numpy.typing import NDArray
from scipy.fftpack import dct
from scipy.stats import binom

logger = logging.getLogger(__name__)


def conv(img: NDArray, kernel: NDArray) -> NDArray:
    """
    Performs a 2D convolution between an image and a kernel. The output is cropped
    to remove the padding induced by the kernel, ensuring the output size matches
    the input image size.

    Args:
        img (NDArray): Input image as a 2D array.
        kernel (NDArray): Convolution kernel as a 2D array.

    Returns:
        NDArray: Convolved image cropped to match the input image dimensions.
    """

    C = cv2.filter2D(img, -1, kernel)
    H = np.floor(np.array(kernel.shape) / 2).astype(int)
    C = (
        C[H[0] : -H[0] + 1, H[1] : -H[1] + 1]
        if kernel.shape[0] % 2 == 0
        else C[H[0] : -H[0], H[1] : -H[1]]
    )
    return C


def valid_blocks_0(img: NDArray, w: int) -> NDArray:
    """
    Computes a mask of valid blocks (not containing saturated pixels).

    mask[i,j] = 1 means that [i,j] is a valid block origin,
    mask[i,j] = 0 menas than the block with origin at [i,j] contains at
    least one saturated pixel, and therefore it is masked as non-valid.

    See Alg. 1 in the paper

    Args:
        img (NDArray): Input image.
        w (int): Side length of the square blocks.

    Returns:
        NDArray: Binary mask where 1 are valid blocks.
    """
    img_not_saturated = np.ones((img.shape[0], img.shape[1]))
    for ch in range(img.shape[2]):
        aux_0 = img[:, :, ch] < img[:, :, ch].max()
        aux_1 = img[:, :, ch] > img[:, :, ch].min()
        aux = aux_0 * aux_1
        img_not_saturated *= aux
    kernel = np.ones((w, w))
    img_int = conv(img_not_saturated.astype(np.uint8), kernel)
    mask = img_int > w**2 - 0.5
    return mask


def compute_valid_blocks_indices(img: NDArray, w: int) -> NDArray:
    """
    Computes the indices of the valid blocks (not containing saturated
    pixels).

    Args:
        img (NDArray): Image to compute indices on.
        w (int): Square block size.

    Return:
        NDArray: Indices of the valid blocks
    """
    valid_blocks_mask = valid_blocks_0(img, w)
    blocks_list = valid_blocks_mask.reshape(-1)
    valid_blocks_indices = np.where(blocks_list == 1)[0]
    return valid_blocks_indices


def all_image_means(img: NDArray, w: int) -> NDArray:
    """
    Computes the means for all the wxw blocks in the image.
    See Alg. 2 in the paper

    img_means[i,j,ch] is the mean of the wxw block with origin [i,j] for
    channel ch.

    Args:
        img (NDArray): Image to compute the means on.
        w (int): Side length of the square blocks.

    Returns:
        NDArray: 3D array containing the mean values of each block for each channel.
    """
    kernel = (1 / w**2) * np.ones((w, w))
    img_means = conv(img, kernel)
    if img.shape[2] == 1:
        return img_means.reshape(img_means.shape[0], img_means.shape[1], 1)
    else:
        return img_means


def sort_blocks_means(
    ch: int, img_means: NDArray, valid_blocks_indices: NDArray
) -> NDArray:
    """
    Sorts valid blocks indices based on their mean intensity in a specific channel.
    See Alg 3. in the paper

    Args:
        ch (int): Index of the channel.
        img_means (NDArray): 3D array containing mean values for each block and channel.
        valid_blocks_indices (NDArray): Indices of valid blocks.

    Returns:
        NDArray: Sorted indices of valid blocks based on mean intensity in the specified
        channel.
    """
    means_aux = img_means[:, :, ch].reshape(-1)
    means = means_aux[valid_blocks_indices]
    valid_blocks_aux = valid_blocks_indices[np.argsort(means)]
    return valid_blocks_aux


def get_T(w: int) -> int:
    """
    Returns the threshold for defining low-medium frequencies based on block size.
    See Alg. 6 in the paper

    Args:
        w (int): Side length of the square blocks.

    Returns:
        int: Threshold to separate low and medium frequencies.
    """
    if w == 3:
        return 3
    elif w == 5:
        return 5
    elif w == 7:
        return 8
    elif w == 8:
        return 9
    else:
        logger.warning(f"Unknown block side {w}")
    return 0


def get_T_mask(w: int) -> NDArray:
    """
    Computes a mask for low-medium frequency components based on block size.
    See Alg. 6 in the paper

    Args:
        w (int): Side length of the square blocks.

    Returns:
        NDArray: Binary mask indicating low-medium frequency components.
    """
    mask = np.zeros((w, w))
    th = get_T(w)

    for i in range(w):
        for j in range(w):
            if (0 != i + j) and (i + j < th):
                mask[i, j] = 1
    return mask


def DCT_all_blocks(img_blocks: NDArray) -> NDArray:
    """
    Computes the DCT II of all the wxw overlapping blocks in the image.

    Args:
        img_blocks (NDArray): Input image blocks.

    Returns:
        NDArray: DCT II of all image blocks.
    """
    return dct(dct(img_blocks, axis=1, norm="ortho"), axis=2, norm="ortho")


def low_freq_DCT(DCTS: NDArray, T_mask: NDArray) -> NDArray:
    """
    Masks the high-frequency coefficients (given by T_mask) of the DCT II
    coefficients in DCTS.

    Args:
        DCTS (NDArray): DCT II coefficients of all blocks.
        T_mask (NDArray): Binary mask indicating low-medium frequency components.

    Returns:
        NDArray: Masked DCT II coefficients.
    """
    tile_T_mask = np.tile(T_mask, (DCTS.shape[0], 1, 1))
    return np.multiply(DCTS.astype(np.float32), tile_T_mask.astype(np.float32))


def compute_low_freq_var(img_blocks: NDArray, w: int) -> NDArray:
    """
    Computes the variance of the low-med frequencies of the DCT coefficients
    (given by T_mask), on the wxw blocks given in img_blocks.
    See Alg. 7 in the paper

    Args:
        img_blocks (NDArray): Input image blocks.
        w (int): Side length of the square blocks.

    Returns:
        NDArray: Array of low-med frequencies variance for all image blocks.
    """
    DCTS = DCT_all_blocks(img_blocks)
    T_mask = get_T_mask(w)
    LF = low_freq_DCT(DCTS, T_mask)
    LF_2 = LF**2
    VL = np.sum(LF_2, axis=(1, 2))
    return VL


def update_samples_per_bin(b: int, num_blocks: int) -> int:
    """
    Updates the number of samples per bin so that each bin has roughly the
    same amount of blocks.
    See Alg. 4 in the paper

    Args:
        b (int): Number of samples per bin.
        num_blocks (int): Total number of blocks.

    Returns:
        int: Updated number of samples per bin.
    """
    num_bins = int(round(num_blocks / b))
    if num_bins == 0:
        num_bins = 1
    b_updated = int(num_blocks / num_bins)
    return b_updated


def bin_block_list(
    num_bins: int, Bin: int, blocks_sorted_means: NDArray, b: int
) -> NDArray:
    """
    Computes the list of blocks corresponding to bin Bin, having b elements
    The last Bin might have more elements.
    See Alg. 5 in the paper

    Args:
        num_bins (int): Number of bins.
        Bin (int): Index of the bin.
        blocks_sorted_means (NDArray): Blocks sorted by mean intensity.
        b (int): Number of samples per bin.

    Returns:
        NDArray: Array of blocks in the given bin.
    """
    num_blocks = len(blocks_sorted_means)

    if Bin == num_bins - 1:
        bin_list = blocks_sorted_means[int(num_bins - 1) * b : num_blocks]
    else:
        bin_list = blocks_sorted_means[Bin * b : (Bin + 1) * b]
    return bin_list


def select_blocks_VL(
    b: int, n: float, low_freq_var: NDArray, blocks_in_bin: NDArray
) -> NDArray:
    """
    Selects the n percentile of blocks in blocks_in_bin having the lowest
    low_freq_var.
    See Alg. 8 in the paper

    Args:
        b (int): Number of samples per bin.
        n (float): Percentile.
        low_freq_var (NDArray): Array of low-med frequencies variance for all
            image blocks.
        blocks_in_bin (NDArray): Array of blocks in the given bin.

    Returns:
        NDArray: Array of blocks in the given bin, sorted according to their variance.
    """
    VL_bin = low_freq_var[blocks_in_bin]
    N = int(b * n)
    sorted_blocks = blocks_in_bin[np.argsort(VL_bin)][0:N]
    return sorted_blocks


def std_blocks(img_blocks: NDArray, select_low_freq_var: NDArray) -> NDArray:
    """
    Sorts the blocks in select_low_freq_var according to their std.
    See Alg. 10 in the paper

    Args:
        img_blocks (NDArray): Input image blocks.
        select_low_freq_var (NDArray): Array of blocks having the lowest variance in
            low-med frequencies.

    Returns:
        NDArray: Array of blocks with the lowest variance in low-med frequencies
            sorted according to their std.
    """
    stds_blocks = np.std(img_blocks[select_low_freq_var], axis=(1, 2))
    stds_sorted_blocks = np.array(select_low_freq_var)[np.argsort(stds_blocks)]
    return stds_sorted_blocks


def bin_is_valid(b: int, n: float, m: float, stds_sorted_blocks: NDArray) -> bool:
    """
    Check if the bin is valid: that is, if the number of flat blocks
    (blocks having standard deviation equal to zero) are more than the
    m percentile.
    See Alg. 9 in the paper

    Args:
        b (int): Number of samples per bin.
        n (float): Percentile for selecting blocks based on the lowest energy in
            low frequencies.
        m (float): Percentile for selecting blocks based on the lowest standard
            deviation.
        stds_sorted_blocks (NDArray): Array of blocks with the lowest variance in
            low-med frequencies sorted according to their std.

    Returns:
        bool: True if the bin is valid, False if not.
    """
    M = int(b * n * m)
    if len(np.where(stds_sorted_blocks == 0)[0]) < M:
        return True
    else:
        return False


def compute_neighbour_blocks(
    index: List, img: NDArray, W: int
) -> List[Tuple[int, int]]:
    """
    Given a cell (index), compute the neighbour cells according to the
    4- connectivity criteria.

    Args:
        index (List): Index of the cell.
        img (NDArray): Input image.
        W (int): Cell side length.

    Returns:
        List[Tuple[int, int]]: List of neighbour cells.
    """
    neighbours = []
    if index[0] >= 1 and index[0] < int(
        img.shape[0] / W
    ):  # index[0] in range(1,int(img.shape[0] / W)):
        neighbours.append([index[0] + 1, index[1]])
        neighbours.append([index[0] - 1, index[1]])
    if index[0] == 0:
        neighbours.append([index[0] + 1, index[1]])
    if index[0] == int(img.shape[0] / W):
        neighbours.append([index[0] - 1, index[1]])
    if index[1] >= 1 and index[1] < int(
        img.shape[1] / W
    ):  # index[1] in range(1,int(img.shape[1] / W)):
        neighbours.append([index[0], index[1] + 1])
        neighbours.append([index[0], index[1] - 1])
    if index[1] == 0:
        neighbours.append([index[0], index[1] + 1])
    if index[1] == int(img.shape[1] / W):
        neighbours.append([index[0], index[1] - 1])
    return neighbours


def bin_prob(k: int, n: int, p: float) -> float:
    """
    Computes the binomial probability P(X = k) where X~ Bin(n, p).

    Args:
        k (int): Number of successes.
        n (int): Number of trials.
        p (float): Probability of success.

    Returns:
        float: Binomial probability.
    """
    arr = mpmath.binomial(n, k)
    pk = mpmath.power(p, k)
    pp = mpmath.power(1 - p, n - k)
    aux = mpmath.fmul(pk, pp)
    bp = mpmath.fmul(arr, aux)
    return bp


def binom_tail(K: int, N: int, w: int, m: float) -> float:
    """
    Computes P(x >= np.floor(K/w**2)) where X~ Bin(np.ceil(N/w**2), m).

    Args:
        K (int): Number of successes.
        N (int): Number of trials.
        w (int): Side length of the square blocks.
        m (float): Probability of success.

    Returns:
        float: Binomial Tail.

    Note:
        If the precision of scipy is not high enough, the computation is done using mpmath library
    (see bin_prob function)
    """
    if 1 - binom.cdf((np.floor(K / w**2)) - 1, np.ceil(N / w**2), m) != 0:
        return 1 - binom.cdf((np.floor(K / w**2)) - 1, np.ceil(N / w**2), m)
    else:
        list_probs = [
            bin_prob(x, int(np.ceil(N / w**2)), m)
            for x in range(int(np.floor(K / w**2)))
        ]
        cdf = sum(list_probs)
        cdf = min(cdf, 1)
        return 1 - cdf


def NFA(img: NDArray, R: int, K: int, N: int, w: int, m: float, W: int) -> float:
    """
    Computes the NFA of a region R.

    Args:
        img (NDArray): Image subject to analysis.
        R (int): Region size.
        K (int): Number of successes.
        N (int): Number of trials.
        w (int): Side length of the square blocks.
        m (float): Probability of success.
        W (int): Cell size for NFA computation.

    Returns:
        float: NFA.
    """
    Nx, Ny = img.shape[:2]
    PR = (0.316915 / R) * (4.062570**R)
    N_tests = 0.5 * (w**2) * (Nx * Ny / W**2) ** 2 * PR
    return N_tests * binom_tail(K, N, w, m)


def seed_crit_satisfied(
    i: int, j: int, all_blocks: NDArray, red_blocks: NDArray, m: float, mask: NDArray
) -> bool:
    """
    Checks if the cell at (i,j) satisfies the seed criteria, meaning that
    the proportion of blocks in L with respect to the blocks in V is bigger
    than m and that the cell has not already been detected.

    Args:
        i (int): Row index.
        j (int): Column index.
        all_blocks (NDArray): Array of all blocks.
        red_blocks (NDArray): Array of red blocks.
        m (float): Probability of success.
        mask (NDArray): Binary mask indicating valid blocks.

    Returns:
        bool: True if the cell satisfies the seed criteria, False if not.
    """
    if all_blocks[i, j] > 0:
        crit_seed = red_blocks[i, j] / all_blocks[i, j] - m
    else:
        crit_seed = 0
    if crit_seed > 0 and mask[i, j] == 0:
        return True
    else:
        return False


def growing_crit_satisfied(
    all_blocks: NDArray,
    red_blocks: NDArray,
    neighbour: NDArray,
    K_R: int,
    N_R: int,
    w: int,
    m: float,
    R_fin: int,
) -> bool:
    """
    Checks if a neighbour cell satisfies the growing critera to be added
    to R, namely, if it improves the NFA value of the region.

    Args:
        all_blocks (NDArray): Array of all blocks.
        red_blocks (NDArray): Array of red blocks.
        neighbour (NDArray): Neighbour cell.
        K_R (int): Number of successes.
        N_R (int): Number of trials.
        w (int): Side length of the square blocks.
        m (float): Probability of success.
        R_fin (int): Final region size.

    Returns:
        bool: True if the neighbour cell satisfies the growing criteria, False if not.
    """
    N_B = all_blocks[neighbour[0], neighbour[1]]
    K_B = red_blocks[neighbour[0], neighbour[1]]
    if binom_tail(K_R, N_R, w, m) / R_fin > 4 * binom_tail(
        K_R + K_B, N_R + N_B, w, m
    ) / (R_fin + 1):
        return True
    else:
        return False


def compute_NFA(
    img: NDArray, w: int, W: int, m: float, all_blocks: NDArray, red_blocks: NDArray
) -> NDArray:
    """
    Computes the NFA on WxW macroblocks.

    Args:
        img (NdArray): Image to run on.
        w (int): Block size.
        W (int): Cell size for NFA computation.
        m (float): Percentile of blocks with the lowest standard deviation.
        all_blocks (NDArray): Array of all blocks.
        red_blocks (NDArray): Array of red blocks.

    Returns:
        NdArray: Mask of the detected cells.
    """
    thresh = 1
    Nx, Ny = img.shape[:2]
    Nx_LR = int(Nx / W) + 1
    Ny_LR = int(Ny / W) + 1
    mask_LR = np.zeros((Nx_LR, Ny_LR))

    for i in range(Nx_LR):
        for j in range(Ny_LR):
            if seed_crit_satisfied(i, j, all_blocks, red_blocks, m, mask_LR):
                R = [[i, j]]
                R_init = 1
                R_fin = 0
                N_R = all_blocks[i, j]
                K_R = red_blocks[i, j]
                while R_init != R_fin:
                    R_init = len(R)
                    R_fin = R_init
                    for index in R:
                        neighbours = compute_neighbour_blocks(index, img, W)
                        for neighbour in neighbours:
                            grow_crit = growing_crit_satisfied(
                                all_blocks, red_blocks, neighbour, K_R, N_R, w, m, R_fin
                            )
                            if neighbour not in R and grow_crit:
                                R.append(neighbour)
                                N_R += all_blocks[neighbour[0], neighbour[1]]
                                K_R += red_blocks[neighbour[0], neighbour[1]]
                                R_fin += 1
                NFA_region = NFA(img, R_fin, K_R, N_R, w, m, W)
                if NFA_region < thresh:
                    cells = []
                    for index in R:
                        mask_LR[index[0], index[1]] = 1  # It was 255
                        cells.append([W * index[0], W * index[1]])

    mask_resized = np.zeros((Nx_LR * W, Ny_LR * W))
    for i in range(Nx_LR):
        for j in range(Ny_LR):
            mask_resized[i * W : (i + 1) * W, j * W : (j + 1) * W] = mask_LR[i, j]
    return mask_resized[0:Nx, 0:Ny]


def compute_output(
    img: NDArray, w: int, W: int, m: float, V: List, S: List
) -> Tuple[NDArray, NDArray]:
    """
    Computes the output of the Noisesniffer algorithm, including the binary mask
    indicating suspected tampering areas and the image with suspected areas painted
    for visualization.

    Args:
        img (NDArray): Image subject to analysis.
        w (int): Side length of the square blocks.
        W (int): Cell size for NFA computation.
        m (float): Probability of success.
        V (List): List of all blocks.
        S (List): List of red blocks.

    Returns:
        NDArray: A binary mask indicating suspected areas of tampering.
        NDArray: The image with the lowest std patches painted red.
    """
    Nx, Ny = img.shape[:2]
    all_blocks = np.zeros((Nx // W + 1, Ny // W + 1))
    red_blocks = np.zeros((Nx // W + 1, Ny // W + 1))
    aux = Ny - w + 1
    img_paint = img
    for pos in V:
        pos_x = int(pos) // aux
        pos_y = int(pos) % aux
        all_blocks[pos_x // W, pos_y // W] += 1
        img_paint[pos_x : pos_x + w, pos_y : pos_y + w, :] = (255, 255, 255)
    for pos in S:
        pos_x = int(pos) // aux
        pos_y = int(pos) % aux
        red_blocks[pos_x // W, pos_y // W] += 1
        img_paint[pos_x : pos_x + w, pos_y : pos_y + w, :] = (255, 0, 0)
    mask = compute_NFA(img, w, W, m, all_blocks, red_blocks)
    return mask, img_paint
