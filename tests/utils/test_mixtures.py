import numpy as np
import pytest

from photoholmes.utils.clustering.gaussian_mixture import GaussianMixture
from photoholmes.utils.clustering.gaussian_uniform import GaussianUniformEM


# =========================== Guassian Mixture =========================================
class TestGaussianMixture:
    @pytest.fixture
    def gm(self):
        self.n_components = 2
        return GaussianMixture(n_components=self.n_components)

    def test_init(self, gm):
        assert gm.n_components == self.n_components
        assert hasattr(gm, "gm")

    def test_fit(self, gm):
        # Create a numpy array
        feature_shape = (100, 10)
        features = np.random.rand(*feature_shape)

        # Fit the GaussianMixture
        mus, covs = gm.fit(features)

        # Check that the means and covariances have the right shape
        assert mus.shape == (self.n_components, feature_shape[-1])
        assert covs.shape == (self.n_components, feature_shape[-1], feature_shape[-1])


# =========================== Guassian Uniform =========================================
class TestGaussianUniformEM:
    @pytest.fixture
    def gu(self):
        return GaussianUniformEM(n_init=2)

    def test_init(self, gu: GaussianUniformEM):
        assert gu.n_init == 2
        assert hasattr(gu, "pi")

    def test_fit(self, gu: GaussianUniformEM):
        # Create a numpy array
        X1_shape = (100, 10)
        X2_shape = (100, 10)
        X1 = np.random.rand(*X1_shape)
        X2 = np.random.uniform(0, 1, X2_shape)
        features = np.vstack((X1, X2))

        # Fit the GaussianUniformEM
        mean, covariance_matrix, pi = gu.fit(features)

        # Check that the mean, covariance_matrix, and pi have the right shape
        assert mean.shape == (X1_shape[-1],)
        assert covariance_matrix.shape == (X1_shape[-1], X1_shape[-1])
        assert isinstance(pi, float)

    def test_predict(self, gu: GaussianUniformEM):
        # Create a numpy array
        X1_shape = (100, 10)
        X2_shape = (100, 10)
        X1 = np.random.rand(*X1_shape)
        X2 = np.random.uniform(0, 1, X2_shape)
        features = np.vstack((X1, X2))

        # Fit the GaussianUniformEM
        gu.fit(features)

        # Predict the class of the samples
        gammas, mahal = gu.predict(features)

        # Check that the gammas and mahal have the right shape
        assert gammas.shape == (X1_shape[0] + X2_shape[0],)
        assert mahal.shape == (X1_shape[0] + X2_shape[0],)
