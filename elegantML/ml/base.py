"""
Base classes for classical ML models in ElegantML.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np

__all__ = ["BaseModel", "ProbabilisticModel"]


class BaseModel(ABC):
    """Abstract base class for estimators with fit/predict interface."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ProbabilisticModel(BaseModel):
    """Extension for models that expose predictive probabilities."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.log(np.clip(probs, 1e-15, 1.0))
