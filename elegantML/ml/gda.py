"""
Gaussian Discriminant Analysis (skeleton).

Provides an interface-compatible placeholder for GDA so callers can
`import GaussianDiscriminantAnalysis` from `elegantML.ml`. Implementation
can be added later without changing public imports.
"""

import numpy as np
from .base import ProbabilisticModel

__all__ = ["GaussianDiscriminantAnalysis"]


class GaussianDiscriminantAnalysis(ProbabilisticModel):
    """Gaussian Discriminant Analysis."""

    def __init__(self) -> None:
        # Placeholder for any future hyperparameters
        self.phi : np.ndarray | None = None
        self.mu  : np.ndarray | None = None
        self.sigma : np.ndarray | None = None
    def _normal(self, x: np.ndarray) -> np.ndarray:
        """ Returns class conditional density p(x_i | y = c_i) for each x_i in X, under class c_i. """
        B,n_features = x.shape
        det = np.linalg.det(self.sigma)  # shape: (K,)
        norm = 1.0 / np.sqrt((2*np.pi)**n_features*det)
        x_mean = x[None, :, :] - self.mu[:, None, :]  # shape: (K,B,n_features)
        exponent = -0.5 * np.sum(x_mean @ np.linalg.inv(self.sigma) @ x_mean.transpose(0, 2, 1), axis=2)  #(K,B,n) @ (K,n,n) @ (K,n,B) -> (K,B)
        ccd = norm * np.exp(exponent)
        return ccd.T  # shape: (B,K)
    def _hypothesis(self, x: np.ndarray) -> np.ndarray:
        """ Returns the posterior p(y|x_new;theta,X_train), class probability for an input."""
        raw_posterior = self.phi[None, :]*self._normal(x) # shape: (,K)*(B,K) -> (B,K)
        posterior = raw_posterior / np.sum(raw_posterior, axis=1, keepdims=True)
        return posterior 
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianDiscriminantAnalysis":
        B, n_features = X.shape
        self.no_classes = len(np.unique(y))
        indicator = np.eye(self.no_classes)[y]  # shape: (B,K)
        cls_sum = indicator.T @ X  # shape: (K,n_features)
        cls_freq = indicator.sum(axis=0)  # shape: (K,)
        self.phi = cls_freq / B
        self.mu = cls_sum / np.clip(cls_freq[:, None], 1e-8, None)  # shape: (K,n_features)
        # vecotrize the computation of sigma
        self.sigma = np.zeros((self.no_classes, n_features, n_features))
        # Compute covariance matrices for each class with vectorization
        for c in range(self.no_classes):
            X_c = X[y == c]
            x_mean = X_c - self.mu[c]
            sigma_c = (x_mean.T @ x_mean) / cls_freq[c]
            self.sigma[c] = 0.5 * (sigma_c + sigma_c.T)  # to ensure symmetry
        return self
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._hypothesis(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) #argmax is zero indexed
