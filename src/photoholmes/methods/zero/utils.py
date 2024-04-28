import mpmath
import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom


def bin_prob(k: int, n: int, p: float) -> float:
    """
    Computes the binomial probability P(X = k) where X~ Bin(n, p).

    Args:
        k (int): number of successes.
        n (int): number of trials.
        p (float): probability of success.

    Returns:
        float: binomial probability.
    """
    arr = mpmath.binomial(n, k)
    pk = mpmath.power(p, k)
    pp = mpmath.power(1 - p, n - k)
    aux = mpmath.fmul(pk, pp)
    bp = mpmath.fmul(arr, aux)
    return bp


def binom_tail(ks: np.ndarray, n: int, p: float) -> NDArray:
    """
    Computes P(X >= k) where X~ Bin(n, p), for each k in ks.

    Args:
        ks (np.ndarray): array of k values.
        n (int): total amount of independent Bernoulli experiments.
        p (float): probability of success of each Bernoulli experiment.

    Returns:
        NDArray: array of P(X >= k) for each k in ks.
    """
    cdf = binom.cdf(ks, n, p)
    if (cdf != 1).all():
        return 1 - cdf
    else:
        cdf = np.zeros_like(ks)
        for i, k in enumerate(ks):
            cdf[i] = np.sum(np.array([bin_prob(x, n, p) for x in range(int(k))]))
        cdf[cdf > 1] = 1
        return 1 - cdf


def log_bin_tail(ks: NDArray, n: int, p: float) -> NDArray:
    """
    Computes the array of the logarithm of the binomial tail, for an array of k values,
    and two fixed parameters n,p. Computes a light or high-precision version as needed.

    Args:
        ks (NDArray): array of k values.
        n (int): total amount of independent Bernoulli experiments.
        p (float): probability of success of each Bernoulli experiment.

    Returns:
        NDArray: array of the logarithm of the binomial tail for each k in ks.
    """
    cdf = binom.cdf(ks, n, p)
    if (cdf != 1).all():
        return np.log10(1 - cdf)
    else:
        log_bin_tail_array = np.empty_like(ks, dtype=float)
        for i, k in enumerate(ks):
            bin_tail = mpmath.nsum(lambda x: bin_prob(x, n, p), [int(k), int(k) + 50])
            log_bin_tail_array[i] = (
                mpmath.log(bin_tail, 10) if bin_tail > 0 else -np.inf
            )

        return log_bin_tail_array


def log_nfa(N_tests: int, ks: NDArray, n: int, p: float) -> NDArray:
    """
    Computes the array of the logarithm of NFA for a given amount N_tests,
    an array of k values, and two fixed parameters n,p.

    Args:
        N_tests (int): total amount of tests.
        ks (NDArray): array of k values.
        n (int): total amount of independent Bernoulli experiments.
        p (float): probability of success of each Bernoulli experiment.

    Returns:
        NDArray: array of the logarithm of the NFA for each k in ks.
    """
    return np.log10(N_tests) + log_bin_tail(ks, n, p)


def closing(mask: NDArray, W: int = 9) -> NDArray:
    Y, X = mask.shape

    image_out = np.zeros_like(mask)
    mask_aux = np.zeros_like(mask)
    for x in range(W, X - W):
        for y in range(W, Y - W):
            if mask[y, x] != 0:
                mask_aux[y - W : y + W + 1, x - W : x + W + 1] = 1
                image_out[y - W : y + W + 1, x - W : x + W + 1] = 1

    for x in range(W, X - W):
        for y in range(W, Y - W):
            if mask_aux[y, x] == 0:
                image_out[y - W : y + W + 1, x - W : x + W + 1] = 0

    return image_out
