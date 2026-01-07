"""
Neural Network Module
=====================

This module provides building blocks for constructing neural networks from scratch
using NumPy, including layers, activations, optimizers, and loss functions.

Components:
-----------
- layers: Neural network layers (Linear, BatchNorm, Dropout, Sequential)
- activations: Activation functions (ReLU, Sigmoid, Tanh, Softmax, etc.)
- optimizers: Optimization algorithms (BGD, BGDwithMomentum, Adam)
- losses: Loss functions (MSE, BinaryCrossEntropy, CrossEntropyLoss, etc.)
"""

from elegantML.nn.layers import (
    Variable,
    Linear,
    Sequential,
    BatchNorm,
    Dropout
)

from elegantML.nn.activations import (
    Activation,
    Tanh,
    Relu,
    LeakyRelu,
    Sigmoid,
    Softmax
)

from elegantML.nn.optimizers import (
    Optimizer,
    BGD,
    BGDwithMomentum,
    Adam
)

from elegantML.nn.losses import (
    Loss,
    MSE_CostFunction,
    BinaryCrossEntropy,
    CrossEntropyLoss,
    L2regularizer
)

__all__ = [
    # Layers
    'Variable',
    'Linear',
    'Sequential',
    'BatchNorm',
    'Dropout',
    # Activations
    'Activation',
    'Tanh',
    'Relu',
    'LeakyRelu',
    'Sigmoid',
    'Softmax',
    # Optimizers
    'Optimizer',
    'BGD',
    'BGDwithMomentum',
    'Adam',
    # Losses
    'Loss',
    'MSE_CostFunction',
    'BinaryCrossEntropy',
    'CrossEntropyLoss',
    'L2regularizer'
]
