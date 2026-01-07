"""
Ordinary Least Squares Linear Regression.
"""

import numpy as np
from .base import BaseModel

__all__ = ["LinearRegression"]


class LinearRegression(BaseModel):
	"""
	Ordinary least squares with optional intercept.

	Parameters
	----------
	fit_intercept : bool, default True
		Whether to include an intercept term.
	"""

	def __init__(self, fit_intercept: bool = True) -> None:
		self.fit_intercept = fit_intercept
		self.coef_: np.ndarray | None = None
		self.intercept_: float = 0.0

	def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
		X_design = self._add_intercept(X) if self.fit_intercept else X
		# Closed-form solution using pseudo-inverse for stability
		params = np.linalg.pinv(X_design) @ y
		if self.fit_intercept:
			self.intercept_ = float(params[0])
			self.coef_ = params[1:]
		else:
			self.intercept_ = 0.0
			self.coef_ = params
		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		if self.coef_ is None:
			raise RuntimeError("Model is not fitted; call fit() first.")
		return X @ self.coef_ + self.intercept_

	def _add_intercept(self, X: np.ndarray) -> np.ndarray:
		return np.c_[np.ones((X.shape[0], 1)), X]
