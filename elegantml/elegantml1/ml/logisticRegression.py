"""
Logistic Regression
===================

Binary logistic regression with a minimal gradient-descent optimizer
and a clean, NumPy-first API.

Methods
-------
- `fit(X, y)`: optimize parameters by gradient descent
- `predict_proba(X)`: sigmoid probabilities
- `predict(X)`: binary labels via 0.5 threshold
"""

import numpy as np
from .base import ProbabilisticModel

__all__ = ["LogisticRegression"]


class LogisticRegression(ProbabilisticModel):
	"""Binary logistic regression estimator.

	Parameters
	----------
	lr : float, default 0.01
		Learning rate for optimization.
	max_iter : int, default 1000
		Maximum optimization iterations.
	fit_intercept : bool, default True
		Whether to include an intercept term.
	eps : float, default 1e-15
		Numerical stability constant for log-likelihood.

	Attributes
	----------
	coef_ : ndarray or None
		Coefficient vector of shape (n_features,) (or including intercept during optimization).
	intercept_ : float
		Intercept term.
	"""

	def __init__(self, lr: float = 0.01, max_iter: int = 1000, fit_intercept: bool = True, eps : float = 1e-15) -> None:
		self.lr = lr
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.coef_: np.ndarray | None = None
		self.intercept_: float = 0.0
		self.eps :float = eps
        
	def _hypothesis(self, X: np.ndarray) -> np.ndarray:
		""" Sigmoid hypothesis function. """
		z = np.dot(X,self.coef_) + self.intercept_
		return 1/(1 + np.clip(np.exp(-z),-500, 500)) #clipping added for numerical stability
	def _logProb(self, X: np.ndarray, y: np.ndarray) -> float:
		""" Compute the log-likelihood of the data under the model. """
		y_hat = self._hypothesis(X)
		Ll = np.sum(y*np.log(y_hat + self.eps) + (1 - y)*np.log(1 - y_hat + self.eps),axis=-1)
		return Ll
	def _add_intercept(self, X: np.ndarray) -> np.ndarray:
		""" Add intercept term to feature matrix if fit_intercept is True. """
		if self.fit_intercept:
			ones = np.ones((X.shape[0], 1))
			return np.hstack([ones, X])
		return X

	def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
		"""Fit the logistic regression model by gradient descent.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Training inputs.
		y : ndarray, shape (n_samples,)
			Binary targets in {0,1}.

		Returns
		-------
		self : LogisticRegression
			Fitted estimator.
		"""
		X = self._add_intercept(X) if self.fit_intercept else X
		B,n_features = X.shape
		self.coef_ = np.random.rand(n_features)
		for iter in range(self.max_iter):
			y_hat = self._hypothesis(X)
			error = y_hat - y
			dJ_dw = np.dot(error,X) / B
			self.coef_ = self.coef_ - self.lr*dJ_dw
		if self.fit_intercept:
			self.intercept_ = self.coef_[0]
			self.coef_ = self.coef_[1:]
		else:
			self.intercept_ = 0.0
			self.coef_ = self.coef_
		return self
	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		"""Predict class probabilities via the sigmoid function.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Input samples.

		Returns
		-------
		P : ndarray, shape (n_samples,)
			Probability of the positive class.
		"""
		X = self._add_intercept(X) if self.fit_intercept else X
		return self._hypothesis(X)

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Predict binary class labels using a 0.5 threshold.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Input samples.

		Returns
		-------
		y_pred : ndarray, shape (n_samples,)
			Predicted class labels in {0,1}.
		"""
		probs = self.predict_proba(X)
		return (probs >= 0.5).astype(int)
