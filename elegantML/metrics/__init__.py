"""
Metrics Module
==============

This module provides metrics to evaluate model performance.

Functions:
----------
- accuracy_score: Calculate accuracy classification score
- precision_score: Calculate precision score
- recall_score: Calculate recall score
- f1_score: Calculate F1 score
- confusion_matrix: Compute confusion matrix
- mean_squared_error: Mean squared error regression loss
- mean_absolute_error: Mean absolute error regression loss
- r2_score: R² (coefficient of determination) regression score
"""

import numpy as np

__all__ = [
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'mean_squared_error',
    'mean_absolute_error',
    'r2_score'
]


# Classification Metrics
def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy classification score.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns
    -------
    score : float
        Accuracy score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns
    -------
    cm : ndarray of shape (n_classes, n_classes)
        Confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return cm


def precision_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Calculate precision score.
    
    Precision = TP / (TP + FP)
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str, default='binary'
        Type of averaging ('binary', 'macro', 'micro')
    pos_label : int, default=1
        The positive class label (for binary classification)
        
    Returns
    -------
    precision : float
        Precision score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    elif average == 'macro':
        classes = np.unique(y_true)
        precisions = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return np.mean(precisions)
    
    elif average == 'micro':
        tp = np.sum(y_true == y_pred)
        fp = np.sum(y_true != y_pred)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Calculate recall score.
    
    Recall = TP / (TP + FN)
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str, default='binary'
        Type of averaging ('binary', 'macro', 'micro')
    pos_label : int, default=1
        The positive class label (for binary classification)
        
    Returns
    -------
    recall : float
        Recall score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    elif average == 'macro':
        classes = np.unique(y_true)
        recalls = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return np.mean(recalls)
    
    elif average == 'micro':
        return accuracy_score(y_true, y_pred)


def f1_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Calculate F1 score.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    average : str, default='binary'
        Type of averaging ('binary', 'macro', 'micro')
    pos_label : int, default=1
        The positive class label (for binary classification)
        
    Returns
    -------
    f1 : float
        F1 score
    """
    precision = precision_score(y_true, y_pred, average=average, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, average=average, pos_label=pos_label)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


# Regression Metrics
def mean_squared_error(y_true, y_pred):
    """
    Calculate mean squared error.
    
    MSE = mean((y_true - y_pred)²)
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    mse : float
        Mean squared error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate mean absolute error.
    
    MAE = mean(|y_true - y_pred|)
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    mae : float
        Mean absolute error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Calculate R² (coefficient of determination) score.
    
    R² = 1 - (SS_res / SS_tot)
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    r2 : float
        R² score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
