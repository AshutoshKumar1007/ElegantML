"""
Optimization Algorithms
======================

NumPy-based optimization algorithms for training neural networks.

Classes
-------
- `Optimizer`: Abstract base class for optimizers
- `BGD`: Batch Gradient Descent
- `BGDwithMomentum`: Gradient descent with momentum and bias correction
- `Adam`: Adaptive Moment Estimation optimizer
"""

import numpy as np 
from typing import Any
from abc import ABC, abstractmethod

#----------------------------------------------------------------------------------------------------------
class Optimizer(ABC):
    """Abstract base class for optimization algorithms.
    
    Parameters
    ----------
    parameters : list
        List of Variable objects containing trainable parameters.
    lr : float, default=0.01
        Learning rate.
        
    Methods
    -------
    step():
        Perform one optimization step (parameter update).
    """
    def __init__(self,parameters,lr = 0.01):
        self.parameters = parameters
        self.lr = lr
    @abstractmethod
    def step(self):
        pass
#----------------------------------------------------------------------------------------------------------
class BGD(Optimizer):
    """Batch Gradient Descent optimizer.
    
    Performs standard gradient descent: w = w - lr * grad.
    """
    def __init__(self, parameters, lr=0.01):
        """Initialize BGD optimizer.
        
        Parameters
        ----------
        parameters : list
            List of Variable objects to optimize.
        lr : float, default=0.01
            Learning rate.
        """
        super().__init__(parameters, lr)
    def step(self):
        """Perform one gradient descent step.
        
        Updates all parameters by subtracting lr * gradient.
        """
        for params in self.parameters:
            params.value -= self.lr*params.grad
#----------------------------------------------------------------------------------------------------------

class BGDwithMomentum(Optimizer):
    """Gradient Descent with momentum and bias correction.
    
    Uses exponentially weighted moving average of gradients with
    bias correction to accelerate convergence.
    """
    def __init__(self, parameters, lr=0.01,momentum = 0.9):     # averaging over 10 grads
        """Initialize momentum optimizer.
        
        Parameters
        ----------
        parameters : list
            List of Variable objects to optimize.
        lr : float, default=0.01
            Learning rate.
        momentum : float, default=0.9
            Momentum coefficient (typically 0.9).
        """
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = [np.zeros_like(params.value) for params in self.parameters]
        self.t = 0
    
    def step(self):
        """Perform one momentum-based gradient descent step.
        
        Updates parameters using bias-corrected momentum.
        """
        self.t += 1
        for i,params in enumerate(self.parameters):
            """ bias correction in exponentialy weighted avg """
            self.velocity[i] = self.momentum*self.velocity[i] + (1 - self.momentum)*params.grad
            vel_unbiased = self.velocity[i]/(1 - self.momentum**self.t)
            params.value -= self.lr*vel_unbiased
#----------------------------------------------------------------------------------------------------------

class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer.
    
    Combines momentum and RMSprop with bias correction for adaptive
    learning rates per parameter.
    """
    def __init__(self, parameters, lr=0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-10):
        """Initialize Adam optimizer.
        
        Parameters
        ----------
        parameters : list
            List of Variable objects to optimize.
        lr : float, default=0.01
            Learning rate.
        beta1 : float, default=0.9
            Exponential decay rate for first moment estimates.
        beta2 : float, default=0.999
            Exponential decay rate for second moment estimates.
        eps : float, default=1e-10
            Small constant for numerical stability.
        """
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps 
        self.velocity = [np.zeros_like(params.value) for params in self.parameters]
        self.square = [np.zeros_like(params.value) for params in self.parameters]
        self.t = 0
    def step(self):
        """Perform one Adam optimization step.
        
        Updates parameters using bias-corrected first and second moment
        estimates for adaptive learning rates.
        """
        self.t += 1
        for i, params in enumerate(self.parameters):
            """ Includes bias correction""" 
            self.velocity[i] = (self.beta1*self.velocity[i] + (1 - self.beta1)*params.grad)
            self.square[i] = (self.beta2*self.square[i] + (1 - self.beta2)*params.grad**2)
            vel_unbiased = self.velocity[i]/(1 - self.beta1**self.t)
            sqr_unbiased = self.square[i]/(1 - self.beta2**self.t)
            params.value -= self.lr*vel_unbiased/np.sqrt(sqr_unbiased + self.eps)
