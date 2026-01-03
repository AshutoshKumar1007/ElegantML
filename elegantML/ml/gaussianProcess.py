"""
Gaussian Process Regression (skeleton).
"""
import numpy as np
from .base import ProbabilisticModel

__all__ = ["GaussianProcessRegressor"]

class GaussianProcessRegressor(ProbabilisticModel):
    """Gaussian Process Regressor with configurable kernel (implementation pending)."""

    def __init__(self, kernel: str | None = "rbf", noise: float = 1e-6) -> None:
        self.kernel = self._rbf_kernel
        self.noise = noise
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
    def _rbf_kernel(self,x1,x2, length_scale : float = 1.0, variance: float = 1.0):
        square_dist = np.sum(((x1[:,None,:] - x2[None,:,:])**2)*(length_scale**-2),axis = 2)
        return variance * np.exp(-0.5 * square_dist)
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessRegressor":
        self.X_train = X
        self.y_train = y
        self.K = self.kernel(self.X_train,self.X_train) + self.noise * np.eye(self.X_train.shape[0])
        self.L = np.linalg.cholesky(self.K)
        z = np.linalg.solve(self.L, self.y_train)
        self.alpha = np.linalg.solve(self.L.T, z)
        return self
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        k_ = self.kernel(self.X_train, X)
        k__ = self.kernel(X,X)
        mu = k_.T @ self.alpha
        v = np.linalg.solve(self.L, k_)
        cov = k__ - v.T @ v
        return mu, cov
    def predict(self, X: np.ndarray) -> np.ndarray:
        
        k_ = self.kernel(self.X_train, X)
        k__ = self.kernel(X,X)
        mu = k_.T @ self.alpha
        return mu
    


    