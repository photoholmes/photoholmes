import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


def histogram_period(histogram: NDArray, alpha: float = 1) -> int:
    """
    Detects the repeating period of a histogram using a weighting factor.

    Args:
        histogram (NDArray): The histogram to evaluate for repeating patterns.
        alpha (float): A weighting factor applied during histogram evaluation.
        Defaults to 1.

    Returns:
        int: The detected period of the histogram.
    """
    Hmax = 0
    period = 1
    k0 = np.argmax(histogram)
    kmin, kmax = 0, len(histogram) - 1

    for p in range(1, kmax // 20):
        imin, imax = np.floor((kmin - k0) / p), np.ceil((kmax - k0) / p)
        i = np.arange(imin, imax, dtype=int)
        H = np.sum(histogram[i * p + k0] ** alpha) / (imax - imin + 1)
        if H > Hmax:
            Hmax = H
            period = p

    return period


def fft_period(histogram: NDArray) -> int:
    """
    Determines the dominant period of a histogram using FFT.

    Args:
        histogram (NDArray): The input histogram to evaluate for repeating patterns.

    Returns:
        int: The detected period of the histogram.
    """
    spectrogram = np.abs(np.fft.fftshift(np.fft.fft(histogram)))
    log_spectrogram = np.log(spectrogram)
    c = len(log_spectrogram) // 2

    peaks = find_peaks(log_spectrogram[c - 1 :], distance=10)[0]
    main_peaks = peaks[log_spectrogram[peaks].argsort()[-1:-5:-1]]

    if len(main_peaks) > 1:
        main_peak = np.sort(main_peaks)[1]
        period = np.round(len(histogram) / main_peak).astype(int)
    else:
        period = 1

    return period


ZIGZAG = [
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (4, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 4),
    (0, 5),
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (5, 0),
    (6, 0),
    (5, 1),
    (4, 2),
    (3, 3),
    (2, 4),
    (1, 5),
    (0, 6),
    (0, 7),
    (1, 6),
    (2, 5),
    (3, 4),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (7, 1),
    (6, 2),
    (5, 3),
    (4, 4),
    (3, 5),
    (2, 6),
    (1, 7),
    (2, 7),
    (3, 6),
    (4, 5),
    (5, 4),
    (6, 3),
    (7, 2),
    (7, 3),
    (6, 4),
    (5, 5),
    (4, 6),
    (3, 7),
    (4, 7),
    (5, 6),
    (6, 5),
    (7, 4),
    (7, 5),
    (6, 6),
    (5, 7),
    (6, 7),
    (7, 6),
    (7, 7),
]
