# losses.py
import numpy as np 
from typing import Any
from abc import ABC, abstractmethod

#----------------------------------------------------------------------------------------------------------

class Loss(ABC):
    """Base class for the loss functions."""
    @abstractmethod
    def __call__(self, ypred, ytarget) -> Any:
        pass
    @abstractmethod
    def backward(self, ypred, ytarget):
        pass

#----------------------------------------------------------------------------------------------------------

class MSE_CostFunction(Loss):
    def __call__(self,ypred,ytarget):
        self.ypred = ypred
        self.ytarget = ytarget
        return np.mean((ytarget - ypred)**2)
    def backward(self):
        return (2*(self.ypred - self.ytarget))/self.ytarget.shape[0]


class BinaryCrossEntropy(Loss):
    def __call__(self,ypred,ytarget, eps = 1e-15):
        self.ypred = np.clip(ypred,eps,1-eps) # numerical stability clips the values within the bound
        self.ytarget = ytarget
        return -np.mean(self.ytarget*np.log(self.ypred) + (1 - self.ytarget)*np.log(1 - self.ypred))    
    def backward(self):
        batch_size = self.ytarget.shape[0]
        return -((self.ytarget/self.ypred)  - (1 - self.ytarget)/(1 - self.ypred))/batch_size


#  cover- ups the logic for softmax derivate for now 
class CrossEntropyLoss(Loss):
    def __call__(self,ypred,ytarget,eps = 1e-15):
        self.ypred = np.clip(ypred,eps,None)
        self.ytarget = ytarget
        return -np.mean(np.sum(self.ytarget*np.log(self.ypred),axis = 1))
    def backward(self):
        batch_size = self.ytarget.shape[0]
        return (self.ypred - self.ytarget )/batch_size


class L2regularizer(Loss):
    def __init__(self,model,L2_lambda = 0.0001):
        self.model = model
        self.L2lambda = L2_lambda
    
    def __call__(self):
        self.L2loss = 0
        for layer in self.model.layers:
            if hasattr(layer,'weight'):
                self.L2loss += (self.L2lambda/2)*np.sum(layer.weight.value**2)
        return self.L2loss
    def backward(self):
        # Import here to avoid circular dependency
        from elegantML.nn.layers import Linear
        for layer in self.model.layers:
            if isinstance(layer,Linear):
                layer.weight.grad += self.L2lambda*layer.weight.value
