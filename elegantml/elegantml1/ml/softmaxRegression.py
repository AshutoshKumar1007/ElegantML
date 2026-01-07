"""
Softmax Regression
==================

Multinomial logistic regression (softmax) for multiclass classification
with a minimal, NumPy-first implementation.

Methods
-------
- `fit(X, y)`: optimize class-wise parameters via gradient ascent
- `predict_proba(X)`: class probability distribution (softmax)
- `predict(X)`: class labels via `argmax`
"""

import numpy as np
from .base import ProbabilisticModel

__all__ = ["SoftmaxRegression"]


class SoftmaxRegression(ProbabilisticModel):
	"""Multinomial logistic regression (softmax).

	Parameters
	----------
	lr : float, default 0.01
		Learning rate for optimization.
	max_iter : int, default 1000
		Maximum optimization iterations.
	fit_intercept : bool, default True
		Whether to include an intercept term.

	Attributes
	----------
	coef_ : ndarray or None
		Weight matrix of shape (n_classes, n_features).
	intercept_ : ndarray or None
		Intercept vector of shape (n_classes,).
	"""

	def __init__(self, lr: float = 0.01, max_iter: int = 1000, fit_intercept: bool = True) -> None:
		self.lr = lr
		self.max_iter = max_iter
		self.fit_intercept = fit_intercept
		self.coef_: np.ndarray | None = None
		self.intercept_: np.ndarray | None = None
	def _hypothesis(self, X: np.ndarray) -> np.ndarray:
		""" Softmax hypothesis function. """
		z = np.dot(X,self.coef_) + self.intercept_	
		exp_z = np.exp(z - np.max(z, axis=1,keepdims=True))  # for numerical stability
		return exp_z / np.sum(exp_z, axis=1, keepdims=True)
	def _logProb(self, X: np.ndarray, y: np.ndarray) -> float:
		""" Compute the log-likelihood of the data under the model. """
		y_hat = self._hypothesis(X)
		epsilon = 1e-15
		Ll = np.sum(np.log(y_hat[np.arange(len(y)), y] + epsilon))
		return Ll
	def _add_intercept(self, X: np.ndarray) -> np.ndarray:
		""" Add intercept term to feature matrix if fit_intercept is True. """
		if self.fit_intercept:
			ones = np.ones((X.shape[0], 1))
			return np.hstack([ones, X])
		return X
	def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
		"""Fit the softmax regression model.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Training inputs.
		y : ndarray, shape (n_samples,)
			Integer class labels in [0, K-1].

		Returns
		-------
		self : SoftmaxRegression
			Fitted estimator.
		"""
		X = self._add_intercept(X) if self.fit_intercept else X
		B,n_features = X.shape
		self.no_classes = len(np.unique(y))# count unique classes in y
		self.coef_ = np.random.rand(self.no_classes, n_features) # shape (k,n_feat)
		self.intercept_ = np.random.rand(self.no_classes) # shape (k,)  
		for iter in range(self.max_iter):
			y_hat = self._hypothesis(X) # shape (B,k)
			indicator = np.eye(self.no_classes)[y] # shape (B,k)
			error = indicator - y_hat  # shape (B,k)
			dJ_dw = np.dot(error,X) /B
			self.coef_ = self.coef_ + self.lr*dJ_dw
		if self.fit_intercept:
			self.intercept_ = self.coef_[:,0]
			self.coef_ = self.coef_[:,1:]
		else:
			self.intercept_ = np.zeros(self.no_classes)
		return self
	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		"""Predict class probability distribution via softmax.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Input samples.

		Returns
		-------
		P : ndarray, shape (n_samples, n_classes)
			Class probabilities per sample.
		"""
		X = self._add_intercept(X) if self.fit_intercept else X
		return self._hypothesis(X)
	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Predict class labels via maximum probability.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Input samples.

		Returns
		-------
		y_pred : ndarray, shape (n_samples,)
			Predicted class labels.
		"""
		probs = self.predict_proba(X)
		return np.argmax(probs, axis=1)
