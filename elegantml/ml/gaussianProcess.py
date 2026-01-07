"""
Gaussian Process Regression
===========================

Minimal, NumPy-based Gaussian Process (GP) regressor with an RBF kernel.

This implementation focuses on clarity and a small, readable API:

- `fit(X, y)`: learns the GP posterior given training data
- `predict(X)`: returns the predictive mean at test points
- `predict_proba(X)`: returns both predictive mean and covariance

Notes
-----
- Currently supports the RBF kernel; future kernels can be added.
- Uses a Cholesky factorization for numerical stability.
"""
import numpy as np
from typing import Optional, Tuple
from .base import ProbabilisticModel

__all__ = ["GaussianProcessRegressor"]

class GaussianProcessRegressor(ProbabilisticModel):
    """Gaussian Process regressor with an RBF kernel.

    Parameters
    ----------
    kernel : str, optional
        Kernel name. Currently only "rbf" is supported.
    noise : float, default=1e-6
        Observation noise variance added to the kernel diagonal.

    Attributes
    ----------
    X_train : ndarray or None
        Training inputs of shape (n_train, n_features).
    y_train : ndarray or None
        Training targets of shape (n_train,).
    K : ndarray or None
        Training kernel matrix of shape (n_train, n_train).
    L : ndarray or None
        Cholesky factor of `K` (lower-triangular), shape (n_train, n_train).
    alpha : ndarray or None
        Solution to `K^{-1} y`, used for fast mean predictions.
    """

    def __init__(self, kernel: Optional[str] = "rbf", noise: float = 1e-6) -> None:
        """Initialize the Gaussian Process regressor.

        Notes
        -----
        - The `kernel` parameter is reserved for future extension; the
          implementation currently uses the RBF kernel.
        """
        self.kernel = self._rbf_kernel
        self.noise = noise
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.L: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray,
                    length_scale: float = 1.0, variance: float = 1.0) -> np.ndarray:
        """Radial Basis Function (RBF) kernel.

        Parameters
        ----------
        x1 : ndarray, shape (n1, d)
            First set of input points.
        x2 : ndarray, shape (n2, d)
            Second set of input points.
        length_scale : float, default=1.0
            Kernel length scale parameter.
        variance : float, default=1.0
            Output variance parameter.

        Returns
        -------
        K : ndarray, shape (n1, n2)
            Kernel matrix between `x1` and `x2`.
        """
        square_dist = np.sum(((x1[:, None, :] - x2[None, :, :]) ** 2) * (length_scale ** -2), axis=2)
        return variance * np.exp(-0.5 * square_dist)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        """Fit the GP model.

        Parameters
        ----------
        X : ndarray, shape (n_train, n_features)
            Training inputs.
        y : ndarray, shape (n_train,)
            Training targets.

        Returns
        -------
        self : GaussianProcessRegressor
            Fitted estimator.
        """
        self.X_train = X
        self.y_train = y
        self.K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(self.X_train.shape[0])
        self.L = np.linalg.cholesky(self.K)
        z = np.linalg.solve(self.L, self.y_train)
        self.alpha = np.linalg.solve(self.L.T, z)
        return self

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and covariance at test points.

        Parameters
        ----------
        X : ndarray, shape (n_test, n_features)
            Test inputs where predictions are requested.

        Returns
        -------
        mean : ndarray, shape (n_test,)
            Predictive mean.
        cov : ndarray, shape (n_test, n_test)
            Predictive covariance matrix.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model is not fitted; call fit() first.")

        # Cross-kernel between train and test, and test-test kernel
        k_train_test = self.kernel(X,self.X_train)
        k_test_test = self.kernel(X, X)

        # Predictive mean
        mu = k_train_test.T @ self.alpha  # shape: (n_test,)

        # Predictive covariance via Cholesky solves
        v = np.linalg.solve(self.L, k_train_test)  # shape: (n_train, n_test)
        cov = k_test_test - v.T @ v                # shape: (n_test, n_test)
        return mu, cov

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean at test points.

        Parameters
        ----------
        X : ndarray, shape (n_test, n_features)
            Test inputs.

        Returns
        -------
        mean : ndarray, shape (n_test,)
            Predictive mean.
        """
        k_train_test = self.kernel(X,self.X_train)
        mu = k_train_test.T @ self.alpha
        return mu
    


    