"""
Activation Functions
===================

NumPy-based activation functions for neural networks with forward
and backward pass implementations.

Classes
-------
- `Activation`: Abstract base class for activation functions
- `Tanh`: Hyperbolic tangent activation
- `Relu`: Rectified Linear Unit
- `LeakyRelu`: Leaky ReLU with configurable slope
- `Sigmoid`: Logistic sigmoid activation
- `Softmax`: Softmax activation for multiclass outputs
"""

import numpy as np
from typing import Any
from abc import ABC, abstractmethod


#----------------------------------------------------------------------------------------------------------

class Activation(ABC):
    """Abstract base class for activation functions.
    
    Methods
    -------
    __call__(x):
        Forward pass computing the activation.
    backward(dAl):
        Backward pass computing gradients.
    parameters():
        Return list of trainable parameters (empty for activations).
    """
    @abstractmethod
    def __call__(self, x) -> Any:
        pass
    @abstractmethod
    def backward(self, dAl):
        pass
    @abstractmethod
    def parameters(self):
        pass

#----------------------------------------------------------------------------------------------------------

class Tanh(Activation):
    """Hyperbolic tangent (tanh) activation function.
    
    Computes tanh(x) element-wise with gradient (1 - tanh²(x)).
    """
    def __call__(self, x):
        """Forward pass.
        
        Parameters
        ----------
        x : ndarray, shape (batch_size, ...)
            Input tensor.
            
        Returns
        -------
        out : ndarray, shape (batch_size, ...)
            Activated output.
        """
        self.out = np.tanh(x)
        return self.out
    def backward(self,dAl):
        """Backward pass.
        
        Parameters
        ----------
        dAl : ndarray
            Gradient from subsequent layer.
            
        Returns
        -------
        gradient : ndarray
            Gradient with respect to input.
        """
        self._der = dAl*(1 - self.out**2)
        return self._der
    def parameters(self):
        """Return trainable parameters (none for activations)."""
        return []
    
#----------------------------------------------------------------------------------------------------------

class Relu(Activation):
    """Rectified Linear Unit (ReLU) activation function.
    
    Computes max(0, x) element-wise with gradient 1 for x > 0, else 0.
    """
    def __call__(self, x):
        """Forward pass.
        
        Parameters
        ----------
        x : ndarray, shape (batch_size, ...)
            Input tensor.
            
        Returns
        -------
        out : ndarray, shape (batch_size, ...)
            Activated output (non-negative).
        """
        self.out = np.maximum(0, x)  # ReLU applies max(0, x) element-wise
        return self.out

    def backward(self, dAl):
        """Backward pass.
        
        Parameters
        ----------
        dAl : ndarray
            Gradient from subsequent layer.
            
        Returns
        -------
        gradient : ndarray
            Gradient with respect to input (1 where x > 0, else 0).
        """
        # Gradient is 1 where x > 0, else 0
        self._der = dAl * (self.out > 0).astype(float)
        return self._der
    def parameters(self):
        """Return trainable parameters (none for activations)."""
        return []
    
#----------------------------------------------------------------------------------------------------------

class LeakyRelu(Activation):
    """Leaky ReLU activation function.
    
    Computes max(x, alpha*x) with configurable negative slope alpha.
    """
    def __call__(self, x,alpha = 0.01):
        """Forward pass.
        
        Parameters
        ----------
        x : ndarray, shape (batch_size, ...)
            Input tensor.
        alpha : float, default=0.01
            Slope for negative values.
            
        Returns
        -------
        out : ndarray, shape (batch_size, ...)
            Activated output.
        """
        self.alpha = alpha
        self.out = np.where(x > 0, x, self.alpha * x)  # Applies leaky behavior
        return self.out
    def backward(self, dAl):
        """Backward pass.
        
        Parameters
        ----------
        dAl : ndarray
            Gradient from subsequent layer.
            
        Returns
        -------
        gradient : ndarray
            Gradient with respect to input (1 where x > 0, else alpha).
        """
        # Gradient is 1 where x > 0, else alpha
        self._der = dAl * np.where(self.out > 0, 1, self.alpha)
        return self._der
    def parameters(self):
        """Return trainable parameters (none for activations)."""
        return []
    
#----------------------------------------------------------------------------------------------------------

class Sigmoid(Activation):
    """Sigmoid (logistic) activation function.
    
    Computes 1 / (1 + exp(-x)) with gradient sigmoid(x) * (1 - sigmoid(x)).
    """
    def __call__(self, x):
        """Forward pass.
        
        Parameters
        ----------
        x : ndarray, shape (batch_size, ...)
            Input tensor.
            
        Returns
        -------
        out : ndarray, shape (batch_size, ...)
            Activated output in range (0, 1).
        """
        self.out = 1 / (1 + np.exp(-x))  # Sigmoid function
        return self.out

    def backward(self, dAl):
        """Backward pass.
        
        Parameters
        ----------
        dAl : ndarray
            Gradient from subsequent layer.
            
        Returns
        -------
        gradient : ndarray
            Gradient with respect to input.
        """
        # Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
        self._der = dAl * self.out * (1 - self.out)
        return self._der

    def parameters(self):
        """Return trainable parameters (none for activations)."""
        return []
    
#----------------------------------------------------------------------------------------------------------

class Softmax(Activation):
    """Softmax activation function for multiclass classification.
    
    Converts logits to a probability distribution. Backward pass is
    simplified when used with cross-entropy loss.
    """
    def __call__(self, x):
        """Forward pass.
        
        Parameters
        ----------
        x : ndarray, shape (batch_size, num_classes)
            Input logits.
            
        Returns
        -------
        out : ndarray, shape (batch_size, num_classes)
            Probability distribution (sums to 1 along axis=1).
        """
        x = x - np.max(x,axis=1,keepdims=True) # shape --> (batch_size,num_classes)
        logits = np.exp(x) 
        self.out = logits/np.sum(logits,axis=1,keepdims=True)
        return self.out
    def backward(self,dAl):
        """Backward pass (simplified for cross-entropy integration).
        
        Parameters
        ----------
        dAl : ndarray
            Gradient from subsequent layer.
            
        Returns
        -------
        gradient : ndarray
            Gradient with respect to input.
        """
        return dAl  # Simplification due to integration with cross-entropy loss
        self._der = dAl*2*(1 - self.out)   # actually 2*(1 - self.out)*self.out but this will be cancelled out by the upcoming divider 
        return self._der
    def parameters(self):
        """Return trainable parameters (none for activations)."""
        return []
