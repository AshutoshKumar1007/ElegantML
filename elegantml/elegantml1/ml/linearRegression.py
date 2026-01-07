"""
Linear Regression
=================

Ordinary least squares (OLS) regression with an optional intercept term
and a simple fit/predict API.

Methods
-------
- `fit(X, y)`: closed-form OLS using pseudo-inverse
- `predict(X)`: linear predictions using learned coefficients
"""

import numpy as np
from .base import BaseModel

__all__ = ["LinearRegression"]


class LinearRegression(BaseModel):
	"""Ordinary least squares regressor.

	Parameters
	----------
	fit_intercept : bool, default True
		Whether to include an intercept term.

	Attributes
	----------
	coef_ : ndarray or None
		Coefficient vector of shape (n_features,).
	intercept_ : float
		Intercept term.
	"""

	def __init__(self, fit_intercept: bool = True) -> None:
		self.fit_intercept = fit_intercept
		self.coef_: np.ndarray | None = None
		self.intercept_: float = 0.0

	def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
		"""Fit the OLS model.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Training inputs.
		y : ndarray, shape (n_samples,)
			Training targets.

		Returns
		-------
		self : LinearRegression
			Fitted estimator.
		"""
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
		"""Predict targets for input samples.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Input samples.

		Returns
		-------
		y_pred : ndarray, shape (n_samples,)
			Predicted targets.
		"""
		if self.coef_ is None:
			raise RuntimeError("Model is not fitted; call fit() first.")
		return X @ self.coef_ + self.intercept_

	def _add_intercept(self, X: np.ndarray) -> np.ndarray:
		return np.c_[np.ones((X.shape[0], 1)), X]
