from typing import List, Union

from numpy.typing import NDArray
from sklearn.decomposition import PCA as sklearn_pca


class PCA:
    """Wrapper to use PCA from scikit-learn library"""

    def __init__(self, n_components: int = 25, whiten: bool = True):
        self.n_components = n_components
        self.pca = sklearn_pca(n_components=n_components, whiten=whiten)

    def fit(self, features: Union[NDArray, List[NDArray]]):
        self.pca.fit(features)

    def transform(self, features: Union[NDArray, List[NDArray]]) -> NDArray:
        return self.pca.transform(features)

    def fit_transform(self, features: Union[NDArray, List[NDArray]]) -> NDArray:
        return self.pca.fit_transform(features)

    def get_covariance(self):
        return self.pca.get_covariance()
