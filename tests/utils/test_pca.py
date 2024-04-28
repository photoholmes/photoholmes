import numpy as np
import pytest

from photoholmes.utils.pca import PCA


class TestPCA:
    @pytest.fixture
    def pca(self):
        return PCA(n_components=2)

    def test_fit(self, pca: PCA):
        """
        Test that the PCA can be fitted.
        """
        # Create a numpy array
        features = np.random.rand(100, 10)

        # Fit the PCA
        pca.fit(features)

        # Check that the PCA has been fitted
        assert hasattr(pca.pca, "components_")

    def test_transform(self, pca: PCA):
        """
        Test that the PCA can be fitted and that features can be transformed.
        """
        # Create a numpy array
        features = np.random.rand(100, 10)

        # Fit the PCA and transform the features
        transformed_features = pca.fit_transform(features)

        # Check that the transformed features have the right shape
        assert transformed_features.shape == (100, 2)

    def test_get_covariance(self, pca: PCA):
        """
        Test that the covariance of the PCA can be retrieved.
        """
        # Create a numpy array
        features = np.random.rand(100, 10)

        # Fit the PCA
        pca.fit(features)

        # Get the covariance of the PCA
        covariance = pca.get_covariance()

        # Check that the covariance has the right shape
        assert covariance.shape == (10, 10)
