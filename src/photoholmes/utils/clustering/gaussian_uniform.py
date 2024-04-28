# Code derived from https://github.com/grip-unina/noiseprint and
# code provided from Quentin Bammey, Marina Gardella and Tina Nikoukhah
import warnings
from typing import Optional, Tuple

import numpy as np
import scipy as sp
from numpy.typing import NDArray


class GaussianUniformEM:
    """Class to perform Gaussian Uniform Expectation Maximization algorithm."""

    def __init__(
        self,
        p_outlier_init: float = 1e-2,
        outlier_nlogl: int = 42,
        tol: float = 1e-5,
        max_iter: int = 100,
        n_init: int = 30,
        seed: Optional[int] = None,
    ) -> None:
        """
        Gaussian Uniform Expectation Maximization algorithm.

        Args:
            p_outlier_init (float): Initial probability of being falsified
            outlier_nlogl (int):  Log-likelihood of being falsified
            tol (float): Tolerance used in a single run of the expectation step
            max_iter (int): Maximum number of iterations in a single run of the expectation step
            n_init (int): Number of iterations of EM to run
            seed (Optional[int]): Random seed for parameter initialization
        """
        self.p_outlier_init = p_outlier_init
        self.outlier_nlogl = outlier_nlogl
        self.pi: float = 1 - p_outlier_init
        self.max_iter = max_iter
        self.tol = tol
        assert n_init > 1, "n_init must be greater than 1"
        self.n_init = n_init
        self.best_loss: float = -np.inf

        self.random_state = np.random.RandomState(seed)
        self.covariance_matrix: NDArray
        self.mean: NDArray

    def fit(self, X: NDArray) -> Tuple[NDArray, NDArray, float]:
        """
        Fit the model to the data.

        Args:
            X (NDArray): The data to fit the model to.

        Returns:
            Tuple[NDArray, NDArray, float]: The mean, covariance matrix, and pi.
        """
        self.mean = np.zeros(X.shape[1])
        self.covariance_matrix = np.eye(X.shape[1])
        save = self.mean, self.covariance_matrix, self.pi

        for _ in range(self.n_init):
            # The following lines are necessary to avoid a overflow when the
            # maximization step is performed and the algorithm fails to converge.
            # In case this happens, the run is discarded and the algorithm tries
            # again.
            try:
                warnings.filterwarnings("error", category=RuntimeWarning)
                loss = self._fit_once(X)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
            except RuntimeWarning:
                loss = -np.inf

            if loss > self.best_loss:
                self.best_loss = loss
                save = self.mean, self.covariance_matrix, self.pi
        self.mean, self.covariance_matrix, self.pi = save
        return self.mean, self.covariance_matrix, self.pi

    def _fit_once(self, X: NDArray) -> float:
        """
        Run a single iteration of the EM algorithm max_iter times or til the
        difference in losses is smaller than tol.

        Args:
            X (NDArray): The data to fit the model to.

        Returns:
            float: The loss.
        """
        n_samples, _ = X.shape
        init_index = self.random_state.randint(
            low=0, high=(n_samples - 1), size=(1,)
        ).squeeze()
        self.mean = X[init_index]
        variance = np.var(X, axis=0)
        variance += np.spacing(variance.max())
        self.covariance_matrix = np.diag(variance)
        self.pi = 1 - self.p_outlier_init
        loss_old = np.inf
        loss = 0.0
        gammas, loss, _ = self._e_step(X)
        for i in range(self.max_iter):
            self._m_step(X, gammas)
            gammas, loss, _ = self._e_step(X)
            loss_diff = loss - loss_old
            if 0 <= loss_diff < self.tol * np.abs(loss):
                break
            loss_old = loss
        return loss

    def _m_step(self, X: NDArray, gammas: NDArray) -> None:
        """
        Maximization step.

        Args:
            X (NDArray): The data to fit the model to.
            gammas (NDArray): The gammas from the E step.
        """
        n_samples, n_features = X.shape
        self.pi = float(np.mean(gammas))
        self.mean = gammas.dot(X) / (n_samples * self.pi)
        Xc = (X - self.mean) * np.sqrt(gammas[:, None])
        self.covariance_matrix = (Xc.T @ Xc) / (n_samples * self.pi) + np.spacing(
            self.covariance_matrix
        ) * np.eye(n_features)

    def _cholesky(self, max_attempts: int = 5) -> NDArray:
        """
        Compute the Cholesky decomposition of the covariance matrix.

        Args:
            max_attempts (int): Maximum number of attempts to make the covariance
                matrix positive definite.

        Returns:
            NDArray: The Cholesky decomposition of the covariance matrix.
        """
        try:
            L = np.linalg.cholesky(self.covariance_matrix)
        except np.linalg.LinAlgError:  # covariance_matrix is not positive definite
            for i in range(max_attempts):
                w, v = sp.linalg.eigh(self.covariance_matrix)
                w = np.maximum(w, np.spacing(w.max()))
                self.covariance_matrix = v @ np.diag(w) @ v.T
                try:
                    L = np.linalg.cholesky(self.covariance_matrix)
                    break
                except np.linalg.LinAlgError:
                    continue
            else:  # if it still fails, raise an error
                raise np.linalg.LinAlgError
        return L

    def _get_nlogl(self, X: NDArray) -> Tuple[float, NDArray]:
        """
        Get log likelihood of pristine class.

        Args:
            X (NDArray): The data to fit the model to.

        Returns:
            Tuple[float, NDArray]: The negative log likelihood and the Mahalanobis
                distance.
        """
        _, n_features = X.shape
        L = self._cholesky()  # covariance_matrix = L@L.T
        D = np.diag(L)
        Xc = X - self.mean
        # Mahalanobis distance is now the L2 norm of L⁻¹ @ Xc.T
        # along the components axis
        Xc_m = np.linalg.solve(L, Xc.T)
        mahalanobis = np.sum(Xc_m**2, axis=0)
        nlogl = 0.5 * (mahalanobis + n_features * np.log(2 * np.pi)) + np.sum(np.log(D))

        return nlogl, mahalanobis

    def _e_step(
        self,
        X: NDArray,
    ) -> Tuple[NDArray, float, NDArray]:
        """
        Run the expectation step.

        Args:
            X (NDArray): The data to fit the model to.

        Returns:
            Tuple[NDArray, float, NDArray]: The gammas, the loss, and the Mahalanobis
                distance.
        """
        nlogl, mahal = self._get_nlogl(X)
        log_gammas_inlier = np.log(self.pi) - nlogl
        log_gammas_outlier = np.log(1 - self.pi) - self.outlier_nlogl
        log_gammas_inlier = log_gammas_inlier[:, None]
        log_gammas_outlier = log_gammas_outlier.repeat(log_gammas_inlier.shape[0])[
            :, None
        ]
        log_gammas = np.append(log_gammas_inlier, log_gammas_outlier, axis=1)
        max_log_likelihood = np.max(log_gammas, axis=1, keepdims=True)
        gammas = np.exp(log_gammas - max_log_likelihood)
        dem = np.sum(gammas, axis=1, keepdims=True)
        gammas /= dem
        loss = np.mean(np.log(dem) + max_log_likelihood)
        return gammas[:, 0], loss, mahal

    def predict(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Predict the class of the samples.

        Args:
            X (NDArray): The data to fit the model to.

        Returns:
            Tuple[NDArray, NDArray]: The gammas and the Mahalanobis distance.
        """
        if self.best_loss == -np.inf:
            return np.zeros(X.shape[0]), np.zeros(X.shape[0])
        gammas, _, mahal = self._e_step(X)
        return gammas, mahal
