"""
ElegantML Supervised Learning
=============================

NumPy-first implementations of classical supervised machine learning
algorithms with simple, readable APIs.

Includes:
- Linear and logistic regression
- Multiclass softmax regression
- Support Vector Machines (classification & regression)
- Gaussian Processes (regression)
- Relevance Vector Machine
- Gaussian Discriminant Analysis
- Decision Tree Regressor
"""

from .base import BaseModel, ProbabilisticModel
from .linearRegression import LinearRegression
from .logisticRegression import LogisticRegression
from .softmaxRegression import SoftmaxRegression
from .svm import SupportVectorClassifier, SupportVectorRegressor
from .gaussianProcess import GaussianProcessRegressor
from .rvm import RelevanceVectorMachine
from .gda import GaussianDiscriminantAnalysis
from .tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = [
	"BaseModel",
	"ProbabilisticModel",
	"LinearRegression",
	"LogisticRegression",
	"SoftmaxRegression",
	"GaussianDiscriminantAnalysis",
	"GaussianProcessRegressor",
	"SupportVectorClassifier",
	"SupportVectorRegressor",
	"RelevanceVectorMachine", #TODO to be implemented.
	"DecisionTreeClassifier",
	"DecisionTreeRegressor", #TODO to be implemented.
]
