"""
ElegantML Base Estimators
=========================

Abstract base classes that define the standard estimator interfaces
used across ElegantML's supervised learning models.

Classes
-------
- `BaseModel`: fit/predict interface for estimators
- `ProbabilisticModel`: extends `BaseModel` with `predict_proba`
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np

__all__ = ["BaseModel", "ProbabilisticModel"]


class BaseModel(ABC):
    """Abstract base estimator.

    Methods
    -------
    fit(X, y):
        Learn model parameters from training data.
    predict(X):
        Predict outputs for input samples.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ProbabilisticModel(BaseModel):
    """Base class for probabilistic estimators.

    Adds support for predictive probabilities/log-probabilities.

    Methods
    -------
    predict_proba(X):
        Return predictive probabilities for input samples.
    predict_log_proba(X):
        Return log-probabilities derived from `predict_proba`.
    """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.log(np.clip(probs, 1e-15, 1.0))
