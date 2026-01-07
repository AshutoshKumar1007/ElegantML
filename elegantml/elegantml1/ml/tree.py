"""
Decision Trees
==============

NumPy-first implementations of Decision Tree algorithms for both
classification and regression tasks with CART-style recursive partitioning.

Methods
-------
- `fit(X, y)`: build tree via recursive splitting
- `predict(X)`: predictions for inputs (classifier: labels; regressor: values)
- `get_tree_depth()`: return depth of learned tree
- `get_n_leaves()`: return number of leaf nodes
"""

import numpy as np
from .base import BaseModel, ProbabilisticModel

__all__ = ["DecisionTreeClassifier", "DecisionTreeRegressor"]


class Node:
	"""Internal tree node representation.
	
	Attributes
	----------
	feature : int or None
		Feature index for splitting.
	threshold : float or None
		Splitting threshold.
	left : Node or None
		Left child node.
	right : Node or None
		Right child node.
	value : float or ndarray or None
		Leaf prediction value.
	"""

	def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
		self.feature = feature
		self.threshold = threshold
		self.left = left
		self.right = right
		self.value = value

	def is_leaf(self):
		"""Check if node is a leaf."""
		return self.value is not None


class DecisionTreeClassifier(BaseModel):
	"""Decision Tree Classifier using CART algorithm.

	Parameters
	----------
	max_depth : int, default=100
		Maximum tree depth. Controls tree complexity.
	min_samples_split : int, default=2
		Minimum samples required to split a node.
	n_features : int or None, default None
		Number of features to consider for splits.
	criterion : {"entropy", "gini"}, default="gini"
		Impurity criterion for split selection.
	categorical_features : iterable of int or None, default None
		Feature indices treated as categorical.

	Attributes
	----------
	root : Node or None
		Root node of the fitted tree.
	classes_ : np.ndarray
		Unique class labels.
	"""

	def __init__(self, max_depth=100, min_samples_split=2, n_features=None, criterion="gini", categorical_features=None):
		if criterion not in ("entropy", "gini"):
			raise ValueError("criterion must be 'entropy' or 'gini'")

		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.n_features = n_features
		self.criterion = criterion
		self.categorical_features = (set(categorical_features) if categorical_features else set())

		self.root = None
		self.classes_ = None

	def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
		"""Build the decision tree classifier.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Training inputs.
		y : ndarray, shape (n_samples,)
			Training class labels.

		Returns
		-------
		self : DecisionTreeClassifier
			Fitted estimator.
		"""
		# Encode labels to integers
		self.classes_, y_encoded = np.unique(y, return_inverse=True)

		self.n_features = (X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1]))

		self.root = self._grow_tree(X, y_encoded, depth=0)
		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""Predict class labels for input samples.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Input samples.

		Returns
		-------
		y_pred : ndarray, shape (n_samples,)
			Predicted class labels.
		"""
		if self.root is None:
			raise RuntimeError("Model is not fitted; call fit() first.")

		encoded_preds = np.array([self._traverse(x, self.root) for x in X])
		return self.classes_[encoded_preds]

	def _grow_tree(self, X, y, depth):
		"""Recursively build the decision tree.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Feature matrix.
		y : ndarray, shape (n_samples,)
			Encoded target labels.
		depth : int
			Current tree depth.

		Returns
		-------
		node : _Node
			Built node (internal or leaf).
		"""
		n_samples, n_features = X.shape
		n_labels = len(np.unique(y))

		if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
			return Node(value=self._most_common_label(y))

		feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

		best_feat, best_val, split_type = self._best_split(X, y, feature_idxs)

		if best_feat is None:
			return Node(value=self._most_common_label(y))

		if split_type == "numerical":
			left_idx, right_idx = self._split_num(X[:, best_feat], best_val)
		else:
			left_idx = np.where(X[:, best_feat] == best_val)[0]
			right_idx = np.where(X[:, best_feat] != best_val)[0]

		left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
		right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

		return Node(feature=best_feat, threshold=best_val, left=left, right=right)

	def _best_split(self, X, y, feature_idxs):
		"""Find the best feature and threshold for splitting.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Feature matrix.
		y : ndarray, shape (n_samples,)
			Target labels.
		feature_idxs : ndarray
			Feature indices to consider.

		Returns
		-------
		best_feat : int or None
			Best feature for splitting.
		best_val : float or None
			Best threshold for splitting.
		split_type : str or None
			Split type ("numerical" or "categorical").
		"""
		best_gain = -1
		best_feat = None
		best_val = None
		split_type = None

		for feat in feature_idxs:
			values = np.unique(X[:, feat])

			if feat in self.categorical_features:
				for v in values:
					gain = self._information_gain_cat(y, X[:, feat], v)
					if gain > best_gain:
						best_gain = gain
						best_feat = feat
						best_val = v
						split_type = "categorical"
			else:
				for v in values:
					gain = self._information_gain_num(y, X[:, feat], v)
					if gain > best_gain:
						best_gain = gain
						best_feat = feat
						best_val = v
						split_type = "numerical"

		return best_feat, best_val, split_type
  
	def _information_gain_num(self, y, X_col, threshold):
		"""Calculate information gain for numerical split.

		Parameters
		----------
		y : ndarray
			Target labels.
		X_col : ndarray
			Feature column values.
		threshold : float
			Splitting threshold.

		Returns
		-------
		gain : float
			Information gain from split.
		"""
		parent = self._impurity(y)
		left, right = self._split_num(X_col, threshold)

		if len(left) == 0 or len(right) == 0:
			return 0

		n = len(y)

		child = (len(left) / n * self._impurity(y[left]) + len(right) / n * self._impurity(y[right]))

		return parent - child

	def _information_gain_cat(self, y, X_col, value):
		"""Calculate information gain for categorical split.

		Parameters
		----------
		y : ndarray
			Target labels.
		X_col : ndarray
			Feature column values.
		value : float or int
			Category value to split on.

		Returns
		-------
		gain : float
			Information gain from split.
		"""
		parent = self._impurity(y)
		left = np.where(X_col == value)[0]
		right = np.where(X_col != value)[0]

		if len(left) == 0 or len(right) == 0:
			return 0

		n = len(y)
		child = (len(left) / n * self._impurity(y[left]) + len(right) / n * self._impurity(y[right]))
		return parent - child

	def _split_num(self, X_col, threshold):
		"""Split samples numerically at threshold.

		Parameters
		----------
		X_col : ndarray
			Feature column values.
		threshold : float
			Splitting threshold.

		Returns
		-------
		left : ndarray
			Indices for X_col <= threshold.
		right : ndarray
			Indices for X_col > threshold.
		"""
		left = np.where(X_col <= threshold)[0]
		right = np.where(X_col > threshold)[0]
		return left, right

	def _impurity(self, y):
		"""Calculate impurity of labels using criterion.

		Parameters
		----------
		y : ndarray
			Target labels.

		Returns
		-------
		impurity : float
			Impurity value.
		"""
		return self._entropy(y) if self.criterion == "entropy" else self._gini(y)

	def _entropy(self, y):
		"""Calculate entropy impurity.

		Parameters
		----------
		y : ndarray
			Target labels.

		Returns
		-------
		entropy : float
			Entropy value.
		"""
		hist = np.bincount(y)
		p = hist / len(y)
		return -np.sum(p[p > 0] * np.log(p[p > 0]))

	def _gini(self, y):
		"""Calculate Gini impurity.

		Parameters
		----------
		y : ndarray
			Target labels.

		Returns
		-------
		gini : float
			Gini impurity value.
		"""
		hist = np.bincount(y)
		p = hist / len(y)
		return 1.0 - np.sum(p ** 2)

	def _most_common_label(self, y):
		"""Get the most common label.

		Parameters
		----------
		y : ndarray
			Target labels.

		Returns
		-------
		label : int
			Most common label.
		"""
		return np.argmax(np.bincount(y))

	def _traverse(self, x, node):
		"""Traverse tree to make prediction for single sample.

		Parameters
		----------
		x : ndarray, shape (n_features,)
			Single input sample.
	node : Node
			Current node in traversal.

		Returns
		-------
		label : int
			Encoded label at leaf.
		"""
		if node.is_leaf():
			return node.value

		if node.feature in self.categorical_features:
			if x[node.feature] == node.threshold:
				return self._traverse(x, node.left)
			return self._traverse(x, node.right)
		else:
			if x[node.feature] <= node.threshold:
				return self._traverse(x, node.left)
			return self._traverse(x, node.right)


class DecisionTreeRegressor(BaseModel):
	"""Decision Tree Regressor using CART algorithm.

	Parameters
	----------
	max_depth : int, default=100
		Maximum tree depth. Controls tree complexity.
	min_samples_split : int, default=2
		Minimum samples required to split a node.
	n_features : int or None, default None
		Number of features to consider for splits.
	criterion : {"mse", "mae"}, default="mse"
		Splitting criterion for variance reduction.
	categorical_features : iterable of int or None, default None
		Feature indices treated as categorical.

	Attributes
	----------
	root : Node or None
		Root node of the fitted tree.
	"""

	def __init__(self, max_depth=100, min_samples_split=2, n_features=None, criterion="mse", categorical_features=None):
		if criterion not in ("mse", "mae"):
			raise ValueError("criterion must be 'mse' or 'mae'")

		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.n_features = n_features
		self.criterion = criterion
		self.categorical_features = (set(categorical_features) if categorical_features else set())

		self.root = None

	def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
		"""Build the decision tree regressor.

		Parameters
		----------
		X : ndarray, shape (n_samples, n_features)
			Training inputs.
		y : ndarray, shape (n_samples,)
			Training targets (continuous values).

		Returns
		-------
		self : DecisionTreeRegressor
			Fitted estimator.
		"""
		# TODO: implement tree building for regression
		pass

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
		# TODO: implement prediction traversal
		pass

