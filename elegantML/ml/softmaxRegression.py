"""
Multiclass softmax (multinomial logistic) regression.
"""

import numpy as np
from .base import ProbabilisticModel

__all__ = ["SoftmaxRegression"]


class SoftmaxRegression(ProbabilisticModel):
	"""
	Multinomial logistic regression using softmax activation.
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
		X = self._add_intercept(X) if self.fit_intercept else X
		return self._hypothesis(X)
	def predict(self, X: np.ndarray) -> np.ndarray:
		probs = self.predict_proba(X)
		return np.argmax(probs, axis=1)
