# activations.py

import numpy as np
import numpy as np 
from typing import Any
from abc import ABC, abstractmethod


#----------------------------------------------------------------------------------------------------------

class Activation(ABC):
    """Base class for the loss functions."""
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
    def __call__(self, x):
        self.out = np.tanh(x)
        return self.out
    def backward(self,dAl):
        self._der = dAl*(1 - self.out**2)
        return self._der
    def parameters(self):
        return []
    
#----------------------------------------------------------------------------------------------------------

class Relu(Activation):
    def __call__(self, x):
        self.out = np.maximum(0, x)  # ReLU applies max(0, x) element-wise
        return self.out

    def backward(self, dAl):
        # Gradient is 1 where x > 0, else 0
        self._der = dAl * (self.out > 0).astype(float)
        return self._der
    def parameters(self):
        return []
    
#----------------------------------------------------------------------------------------------------------

class LeakyRelu(Activation):
    def __call__(self, x,alpha = 0.01):
        self.alpha = alpha
        self.out = np.where(x > 0, x, self.alpha * x)  # Applies leaky behavior
        return self.out
    def backward(self, dAl):
        # Gradient is 1 where x > 0, else alpha
        self._der = dAl * np.where(self.out > 0, 1, self.alpha)
        return self._der
    def parameters(self):
        return []
    
#----------------------------------------------------------------------------------------------------------

class Sigmoid(Activation):
    def __call__(self, x):
        self.out = 1 / (1 + np.exp(-x))  # Sigmoid function
        return self.out

    def backward(self, dAl):
        # Derivative of sigmoid is sigmoid(x) * (1 - sigmoid(x))
        self._der = dAl * self.out * (1 - self.out)
        return self._der

    def parameters(self):
        return []
    
#----------------------------------------------------------------------------------------------------------

class Softmax(Activation):
    def __call__(self, x):
        x = x - np.max(x,axis=1,keepdims=True) # shape --> (batch_size,num_classes)
        logits = np.exp(x) 
        self.out = logits/np.sum(logits,axis=1,keepdims=True)
        return self.out
    def backward(self,dAl):
        return dAl  # Simplification due to integration with cross-entropy loss
        self._der = dAl*2*(1 - self.out)   # actually 2*(1 - self.out)*self.out but this will be cancelled out by the upcoming divider 
        return self._der
    def parameters(self):
        return []
