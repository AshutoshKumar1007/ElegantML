"""
Relevance Vector Machine (skeleton).
"""

import numpy as np
from .base import ProbabilisticModel

__all__ = ["RelevanceVectorMachine"]


class RelevanceVectorMachine(ProbabilisticModel):
	"""RVM for regression/classification (implementation pending)."""

	def __init__(self, kernel: str | None = None, max_iter: int = 500) -> None:
		self.kernel = kernel or "rbf"
		self.max_iter = max_iter
		self.relevance_vectors_: np.ndarray | None = None
  

	def fit(self, X: np.ndarray, y: np.ndarray) -> "RelevanceVectorMachine":
		raise NotImplementedError("RVM.fit is pending implementation.")

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		raise NotImplementedError("RVM.predict_proba is pending implementation.")

	def predict(self, X: np.ndarray) -> np.ndarray:
		raise NotImplementedError("RVM.predict is pending implementation.")
