"""
ElegantML - A NumPy-first Machine Learning Library
===================================================

ElegantML is a NumPy-based machine learning library focused on clear, minimal, 
and mathematically grounded implementations of classical ML and neural networks.

Submodules:
-----------
- nn: Neural network components (layers, activations, optimizers, losses)
- ml: Classical ML algorithms (regression, classification, probabilistic models)
- clustering: Clustering algorithms (K-Means etc.)
- preprocessing: Data preprocessing utilities
- metrics: Model evaluation metrics
"""

__version__ = "0.1.0"
__author__ = "ElegantML Contributors"

# Submodules are available via: from elegantML import ml, nn etc.
# Or: from elegantML.nn import Linear, etc.

from elegantml import ml  # classical ML algorithms
from elegantml import nn  # neural network components
from elegantml import clustering  # clustering algorithms
from elegantml import preprocessing  # data preprocessing utilities
from elegantml import metrics  # model evaluation metrics

__all__ = [
    "__version__",
    "ml",
    "nn",
    "clustering",
    "preprocessing",
    "metrics",
]
