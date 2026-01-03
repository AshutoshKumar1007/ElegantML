"""
Preprocessing Module
====================

This module provides data preprocessing utilities.

Classes:
--------
- StandardScaler: Standardize features by removing mean and scaling to unit variance
- MinMaxScaler: Scale features to a given range
- LabelEncoder: Encode target labels with value between 0 and n_classes-1
- OneHotEncoder: Encode categorical features as one-hot numeric array
"""

import numpy as np

__all__ = ['StandardScaler', 'MinMaxScaler', 'LabelEncoder', 'OneHotEncoder']


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - u) / s
    where u is the mean and s is the standard deviation.
    
    Examples
    --------
    >>> from elegantML.preprocessing import StandardScaler
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> scaler = StandardScaler()
    >>> scaler.fit(X)
    >>> scaler.transform(X)
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X):
        """Compute the mean and std to be used for scaling."""
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1
        return self
    
    def transform(self, X):
        """Perform standardization by centering and scaling."""
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit to data, then transform it."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Scale back the data to the original representation."""
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range.
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data
        
    Examples
    --------
    >>> from elegantML.preprocessing import MinMaxScaler
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> scaler.fit_transform(X)
    """
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        
    def fit(self, X):
        """Compute the minimum and maximum to be used for scaling."""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        
        # Compute scale
        data_range = self.max_ - self.min_
        data_range[data_range == 0] = 1  # Avoid division by zero
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / data_range
        
        return self
    
    def transform(self, X):
        """Scale features to the specified range."""
        X_std = (X - self.min_) * self.scale_
        return X_std + self.feature_range[0]
    
    def fit_transform(self, X):
        """Fit to data, then transform it."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Undo the scaling."""
        X_std = X - self.feature_range[0]
        return X_std / self.scale_ + self.min_


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.
    
    Examples
    --------
    >>> from elegantML.preprocessing import LabelEncoder
    >>> le = LabelEncoder()
    >>> le.fit(['cat', 'dog', 'cat', 'bird'])
    >>> le.transform(['cat', 'bird', 'dog'])
    array([1, 0, 2])
    """
    
    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = None
        
    def fit(self, y):
        """Fit label encoder."""
        self.classes_ = np.unique(y)
        self.class_to_index_ = {label: idx for idx, label in enumerate(self.classes_)}
        return self
    
    def transform(self, y):
        """Transform labels to normalized encoding."""
        return np.array([self.class_to_index_[label] for label in y])
    
    def fit_transform(self, y):
        """Fit label encoder and return encoded labels."""
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        """Transform labels back to original encoding."""
        return np.array([self.classes_[idx] for idx in y])


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    
    Examples
    --------
    >>> from elegantML.preprocessing import OneHotEncoder
    >>> enc = OneHotEncoder()
    >>> X = np.array([[0], [1], [2], [1]])
    >>> enc.fit_transform(X)
    """
    
    def __init__(self):
        self.categories_ = None
        self.n_categories_ = None
        
    def fit(self, X):
        """Fit OneHotEncoder to X."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        self.n_categories_ = [len(cats) for cats in self.categories_]
        return self
    
    def transform(self, X):
        """Transform X using one-hot encoding."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        n_features = sum(self.n_categories_)
        
        X_encoded = np.zeros((n_samples, n_features))
        
        feature_idx = 0
        for i, categories in enumerate(self.categories_):
            for j, category in enumerate(categories):
                mask = X[:, i] == category
                X_encoded[mask, feature_idx + j] = 1
            feature_idx += len(categories)
        
        return X_encoded
    
    def fit_transform(self, X):
        """Fit OneHotEncoder and transform X."""
        self.fit(X)
        return self.transform(X)
