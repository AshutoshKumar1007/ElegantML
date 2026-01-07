"""
Loss Functions
==============

NumPy-based loss functions for neural network training with forward
and backward pass implementations.

Classes
-------
- `Loss`: Abstract base class for loss functions
- `MSE_CostFunction`: Mean Squared Error for regression
- `BinaryCrossEntropy`: Binary cross-entropy for binary classification
- `CrossEntropyLoss`: Cross-entropy for multiclass classification
- `L2regularizer`: L2 (weight decay) regularization
"""

import numpy as np 
from typing import Any
from abc import ABC, abstractmethod

#----------------------------------------------------------------------------------------------------------

class Loss(ABC):
    """Abstract base class for loss functions.
    
    Methods
    -------
    __call__(ypred, ytarget):
        Compute the loss value.
    backward(ypred, ytarget):
        Compute gradient of loss with respect to predictions.
    """
    @abstractmethod
    def __call__(self, ypred, ytarget) -> Any:
        pass
    @abstractmethod
    def backward(self, ypred, ytarget):
        pass

#----------------------------------------------------------------------------------------------------------

class MSE_CostFunction(Loss):
    """Mean Squared Error (MSE) loss function.
    
    Computes mean((y_target - y_pred)²) for regression tasks.
    """
    def __call__(self,ypred,ytarget):
        """Compute MSE loss.
        
        Parameters
        ----------
        ypred : ndarray, shape (batch_size, ...)
            Predicted values.
        ytarget : ndarray, shape (batch_size, ...)
            Ground truth values.
            
        Returns
        -------
        loss : float
            Mean squared error.
        """
        self.ypred = ypred
        self.ytarget = ytarget
        return np.mean((ytarget - ypred)**2)
    def backward(self):
        """Compute gradient of MSE with respect to predictions.
        
        Returns
        -------
        gradient : ndarray
            Gradient of loss with respect to predictions.
        """
        return (2*(self.ypred - self.ytarget))/self.ytarget.shape[0]


class BinaryCrossEntropy(Loss):
    """Binary cross-entropy loss for binary classification.
    
    Computes -mean(y*log(p) + (1-y)*log(1-p)) with numerical stability.
    """
    def __call__(self,ypred,ytarget, eps = 1e-15):
        """Compute binary cross-entropy loss.
        
        Parameters
        ----------
        ypred : ndarray, shape (batch_size, ...)
            Predicted probabilities in (0, 1).
        ytarget : ndarray, shape (batch_size, ...)
            Binary ground truth labels in {0, 1}.
        eps : float, default=1e-15
            Small constant for numerical stability.
            
        Returns
        -------
        loss : float
            Binary cross-entropy loss value.
        """
        self.ypred = np.clip(ypred,eps,1-eps) # numerical stability clips the values within the bound
        self.ytarget = ytarget
        return -np.mean(self.ytarget*np.log(self.ypred) + (1 - self.ytarget)*np.log(1 - self.ypred))    
    def backward(self):
        """Compute gradient of BCE with respect to predictions.
        
        Returns
        -------
        gradient : ndarray
            Gradient of loss with respect to predictions.
        """
        batch_size = self.ytarget.shape[0]
        return -((self.ytarget/self.ypred)  - (1 - self.ytarget)/(1 - self.ypred))/batch_size


#  cover- ups the logic for softmax derivate for now 
class CrossEntropyLoss(Loss):
    """Cross-entropy loss for multiclass classification.
    
    Typically used with softmax activation. Simplified gradient when
    combined with softmax.
    """
    def __call__(self,ypred,ytarget,eps = 1e-15):
        """Compute cross-entropy loss.
        
        Parameters
        ----------
        ypred : ndarray, shape (batch_size, num_classes)
            Predicted class probabilities.
        ytarget : ndarray, shape (batch_size, num_classes)
            One-hot encoded ground truth labels.
        eps : float, default=1e-15
            Small constant for numerical stability.
            
        Returns
        -------
        loss : float
            Cross-entropy loss value.
        """
        self.ypred = np.clip(ypred,eps,None)
        self.ytarget = ytarget
        return -np.mean(np.sum(self.ytarget*np.log(self.ypred),axis = 1))
    def backward(self):
        """Compute gradient of cross-entropy with respect to predictions.
        
        Returns
        -------
        gradient : ndarray
            Gradient of loss with respect to predictions.
        """
        batch_size = self.ytarget.shape[0]
        return (self.ypred - self.ytarget )/batch_size


class L2regularizer(Loss):
    """L2 regularization (weight decay) for neural networks.
    
    Adds penalty term (lambda/2) * sum(W²) to prevent overfitting.
    """
    def __init__(self,model,L2_lambda = 0.0001):
        """Initialize L2 regularizer.
        
        Parameters
        ----------
        model : Sequential
            Neural network model to regularize.
        L2_lambda : float, default=0.0001
            Regularization strength.
        """
        self.model = model
        self.L2lambda = L2_lambda
    
    def __call__(self):
        """Compute L2 regularization penalty.
        
        Returns
        -------
        loss : float
            L2 penalty value.
        """
        self.L2loss = 0
        for layer in self.model.layers:
            if hasattr(layer,'weight'):
                self.L2loss += (self.L2lambda/2)*np.sum(layer.weight.value**2)
        return self.L2loss
    def backward(self):
        """Add L2 gradient to weight gradients.
        
        Modifies weight.grad in-place by adding lambda * weight.
        """
        # Import here to avoid circular dependency
        from elegantml.nn.layers import Linear
        for layer in self.model.layers:
            if isinstance(layer,Linear):
                layer.weight.grad += self.L2lambda*layer.weight.value
