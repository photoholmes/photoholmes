from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture as sklearn_gmm


class GaussianMixture:
    """Wrapper to use Gaussian Mixtures from scikit-learn library"""

    def __init__(self, n_components: int = 2, seed: Union[int, None] = None):
        """
        Gaussian Mixture model.

        Args:
            n_components (int): The number of components in the mixture.
            seed (Union[int, None]): Random seed for parameter initialization.
        """
        self.n_components = n_components
        self.gm = sklearn_gmm(
            n_components=n_components, random_state=np.random.RandomState(seed)
        )

    def fit(self, features: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Predicts masks from a list of images.

        Args:
            features (NDArray): The features to fit the model to.

        Returns:
            Tuple[NDArray, NDArray]: The means and covariances of the model.
        """

        self.gm.fit(features)
        mus = self.gm.means_
        covs = self.gm.covariances_
        mus = np.array(mus)
        covs = np.array(covs)
        return mus, covs
