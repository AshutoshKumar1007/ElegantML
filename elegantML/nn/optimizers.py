# optimizers.py
import numpy as np 
from typing import Any
from abc import ABC, abstractmethod

#----------------------------------------------------------------------------------------------------------
class Optimizer(ABC):
    def __init__(self,parameters,lr = 0.01):
        self.parameters = parameters
        self.lr = lr
    @abstractmethod
    def step(self):
        pass
#----------------------------------------------------------------------------------------------------------
class BGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr)
    def step(self):
        for params in self.parameters:
            params.value -= self.lr*params.grad
#----------------------------------------------------------------------------------------------------------

class BGDwithMomentum(Optimizer):
    def __init__(self, parameters, lr=0.01,momentum = 0.9):     # averaging over 10 grads
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = [np.zeros_like(params.value) for params in self.parameters]
        self.t = 0
    
    def step(self):
        self.t += 1
        for i,params in enumerate(self.parameters):
            """ bias correction in exponentialy weighted avg """
            self.velocity[i] = self.momentum*self.velocity[i] + (1 - self.momentum)*params.grad
            vel_unbiased = self.velocity[i]/(1 - self.momentum**self.t)
            params.value -= self.lr*vel_unbiased
#----------------------------------------------------------------------------------------------------------

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-10):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps 
        self.velocity = [np.zeros_like(params.value) for params in self.parameters]
        self.square = [np.zeros_like(params.value) for params in self.parameters]
        self.t = 0
    def step(self):
        self.t += 1
        for i, params in enumerate(self.parameters):
            """ Includes bias correction""" 
            self.velocity[i] = (self.beta1*self.velocity[i] + (1 - self.beta1)*params.grad)
            self.square[i] = (self.beta2*self.square[i] + (1 - self.beta2)*params.grad**2)
            vel_unbiased = self.velocity[i]/(1 - self.beta1**self.t)
            sqr_unbiased = self.square[i]/(1 - self.beta2**self.t)
            params.value -= self.lr*vel_unbiased/np.sqrt(sqr_unbiased + self.eps)
