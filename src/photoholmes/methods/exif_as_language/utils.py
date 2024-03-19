# Code derived from https://github.com/hellomuffin/exif-as-language
import numpy as np
import scipy
import torch
from numpy.typing import NDArray
from sklearn import cluster
from torch import Tensor


def cosine_similarity(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Cosine similarity between all pairs of vectors in x1 and x2.

    Args:
        x1 (Tensor): Tensor of shape (N, D).
        x2 (Tensor): Tensor of shape (M, D).

    Returns:
        Tensor: Tensor of shape (N, M) with cosine similarity between all pairs of
            vectors.
    """

    x1 = x1 / (x1.norm(dim=-1, keepdim=True) + 1e-32)
    x2 = x2 / (x2.norm(dim=-1, keepdim=True) + 1e-32)

    cos = torch.matmul(x1, x2.t())
    return cos


def normalized_cut(res: NDArray) -> NDArray:
    """
    Spectral clustering via Normalized Cuts

    Args:
        res (NDArray): Affinity matrix between patches.

    Returns:
        NDArray: normalized cut.
    """
    res = 1 - res
    sc = cluster.SpectralClustering(n_clusters=2, n_jobs=-1, affinity="precomputed")
    out = sc.fit_predict(res.reshape((res.shape[0] * res.shape[1], -1)))
    vis = out.reshape((res.shape[0], res.shape[1]))
    return vis


def mean_shift(points_: NDArray, heat_map: NDArray, window: int, iter: int) -> NDArray:
    """
    Applys Mean Shift algorithm in order to obtain a uniform heatmap.

    Args:
        points_ (NDArray): Affinity matrix between patches.
        heat_map (NDArray): Heatmap obtained from the affinity matrix.
        window (int): window size.
        iter (int): number of iterations.

    Returns:
        NDArray: Uniform heatmap after mean shift on rows.
    """
    points = np.copy(points_)
    kdt = scipy.spatial.cKDTree(points)
    eps_5 = np.percentile(
        scipy.spatial.distance.cdist(points, points, metric="euclidean"), window
    )
    if eps_5 != 0:
        try:
            for _ in range(iter):
                for point_ind in range(points.shape[0]):
                    point = points[point_ind]
                    nearest_inds = kdt.query_ball_point(point, r=eps_5)
                    points[point_ind] = np.mean(points[nearest_inds], axis=0)
            val = []
            for i in range(points.shape[0]):
                try:
                    val.append(
                        kdt.count_neighbors(
                            scipy.spatial.cKDTree(np.array([points[i]])), r=eps_5
                        )
                    )
                except ValueError:
                    pass
            ind = np.nonzero(val == np.max(val))
            result = np.mean(points[ind[0]], axis=0).reshape(
                heat_map.shape[0], heat_map.shape[1]
            )
        except ValueError:
            result = np.zeros((heat_map.shape[0], heat_map.shape[1]))
    else:
        result = np.zeros((heat_map.shape[0], heat_map.shape[1]))
    return result
