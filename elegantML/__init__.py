"""
ElegantML - A NumPy-first Machine Learning Library
===================================================

ElegantML is a NumPy-based machine learning library focused on clear, minimal, 
and mathematically grounded implementations of classical ML and neural networks.

Submodules:
-----------
- nn: Neural network components (layers, activations, optimizers, losses)
- linear_models: Classical linear models (Linear Regression, Logistic Regression, etc.)
- tree: Decision trees and ensemble methods
- clustering: Clustering algorithms (K-Means, DBSCAN, etc.)
- preprocessing: Data preprocessing utilities
- metrics: Model evaluation metrics
"""

__version__ = "0.1.0"
__author__ = "ElegantML Contributors"

# Submodules are available via: from elegantML import ml, nn
# Or: from elegantML.nn import Linear, etc.

from elegantML import ml  # classical ML algorithms
from elegantML import nn  # neural network components
from elegantML import clustering  # clustering algorithms

__all__ = [
    "__version__",
    "ml",
    "nn",
    "clustering",
]
