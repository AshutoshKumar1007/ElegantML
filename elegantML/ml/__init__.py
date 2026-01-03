"""
Classical machine learning algorithms for ElegantML.

This subpackage collects NumPy-first implementations and interfaces for
regression, classification, and probabilistic models.
"""

from .base import BaseModel, ProbabilisticModel
from .linearRegression import LinearRegression
from .logisticRegression import LogisticRegression
from .softmaxRegression import SoftmaxRegression
from .svm2 import SupportVectorClassifier, SupportVectorRegressor
from .gaussianProcess import GaussianProcessRegressor
from .rvm import RelevanceVectorMachine
from .gda import GaussianDiscriminantAnalysis
# Reuse tree-based implementations from the sibling tree module
from elegantML.tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = [
	"BaseModel",
	"ProbabilisticModel",
	"LinearRegression",
	"LogisticRegression",
	"SoftmaxRegression",
	"SupportVectorMachine",
	"GaussianProcessRegressor",
	"RelevanceVectorMachine",
	"GaussianDiscriminantAnalysis",
	"DecisionTreeClassifier",
	"DecisionTreeRegressor",
]
