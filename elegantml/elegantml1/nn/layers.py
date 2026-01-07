"""
Neural Network Layers
====================

NumPy-based neural network layer implementations with forward and
backward pass support, model management, and persistence.

Classes
-------
- `Variable`: Wrapper for trainable parameters with value and gradient
- `Linear`: Fully-connected (dense) layer
- `Sequential`: Sequential container for stacking layers
- `BatchNorm`: Batch normalization layer
- `Dropout`: Dropout regularization layer
"""

import numpy as np 
import pandas as pd 
import matplotlib as mpl
from matplotlib import pyplot as plt 
from abc import ABC, abstractmethod
import pickle

#------------------------------------------
class Variable:
    """Wrapper for trainable parameters.
    
    Stores parameter value and its gradient for optimization.
    
    Attributes
    ----------
    value : ndarray
        Parameter value.
    grad : ndarray or scalar
        Gradient with respect to loss.
    """
    def __init__(self, value) -> None:
        self.value = value
        self.grad = 0
      
#----------------------------------------------------------------------------------------------------------  
class Linear:
    """Fully-connected (dense) linear layer.
    
    Performs affine transformation: y = xW + b with optional bias.
    
    Parameters
    ----------
    fan_in : int
        Input dimension.
    fan_out : int
        Output dimension.
    bias : bool, default True
        Whether to include bias term.
        
    Attributes
    ----------
    weight : Variable
        Weight matrix of shape (fan_in, fan_out).
    bias : Variable or None
        Bias vector of shape (1, fan_out).
    """
    def __init__(self,fan_in,fan_out,bias = True) -> None:
        self.weight = Variable(np.random.randn(fan_in, fan_out)/(fan_in)**0.5)  # could have taken fan_out as first dim -> must change the computations acc
        self.bias = Variable(np.random.randn(1,fan_out)) if bias else None
        self.aprev = None
    def __call__(self,x):
        """Forward pass.
        
        Parameters
        ----------
        x : ndarray, shape (batch_size, fan_in)
            Input activations.
            
        Returns
        -------
        out : ndarray, shape (batch_size, fan_out)
            Output activations.
        """
        # Farward pass with batch handling
        self.aprev = x
        self.out = x @ self.weight.value
        if self.bias is not None:
            self.out += self.bias.value
        return self.out
    def backward(self,dzl):
        """Backward pass.
        
        Parameters
        ----------
        dzl : ndarray, shape (batch_size, fan_out)
            Gradient from subsequent layer.
            
        Returns
        -------
        daprev : ndarray, shape (batch_size, fan_in)
            Gradient with respect to input.
        """
        # Backward pass with batch handling
        batch_size = dzl.shape[0]       #! input must have a batch size
        self.weight.grad = (self.aprev.T@dzl)/batch_size  # fan_in * batch_size @ batch_size * fan_out
        if self.bias is not None:
            self.bias.grad = np.sum(dzl,axis=0,keepdims=True) /batch_size  # batch_size*fan_out ----> 1*fanout
        daprev  =  dzl @ self.weight.value.T             #  batch_size * fan_out @ (fan_in * fan_out).T ---> batch_size * fan_in
        return daprev
    def parameters(self):
        """Return list of trainable parameters."""
        return [self.weight] + ([] if self.bias is None else [self.bias]) 

#----------------------------------------------------------------------------------------------------------

class Sequential:
    """Sequential container for neural network layers.
    
    Chains layers together and manages forward/backward passes,
    training/eval modes, and model persistence.
    
    Parameters
    ----------
    layers : list
        List of layer objects (Linear, activations, etc.).
        
    Attributes
    ----------
    layers : list
        Contained layers.
    optimizer : Optimizer or None
        Optimizer for training.
    regulizer : L2regularizer or None
        Regularization handler.
    """
    def __init__(self, layers):
        self.layers = layers
        self.optimizer = None
        self.regulizer = None 
    def train(self):
        """Switch all layers to training mode."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
    def eval(self):
        """Switch all layers to evaluation mode."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
    def __call__(self, x,training = True):
        """Forward pass through all layers.
        
        Parameters
        ----------
        x : ndarray
            Input data.
        training : bool, default True
            Whether in training mode.
            
        Returns
        -------
        out : ndarray
            Output after passing through all layers.
        """
        self.training = training
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    def backprop(self,dypred):
        """Backward pass through all layers.
        
        Parameters
        ----------
        dypred : ndarray
            Gradient of loss with respect to output.
        """
        dAl = dypred
        for layer in reversed(self.layers[1:]):
            dAl = layer.backward(dAl)
        if self.regulizer is not None:
            self.regulizer.backward()
    """  gradient step will be handled by optimizer.step() method call"""
    # def update_params(self,lr = 0.1):
    #     for layer in self.layers:
    #         if hasattr(layer,'weight'):
    #             layer.weight -= lr*layer.dw  # adding the L-2 loss
    #         if hasattr(layer,'bias'):
    #             layer.bias -= lr*layer.db
    #         if isinstance(layer,BatchNorm):
    #             layer.gamma -= lr*layer.dgamma
    #             layer.beta -= lr*layer.dbeta
                 
  
    def parameters(self):
        """Get all trainable parameters from all layers.
        
        Returns
        -------
        params : list
            Flattened list of all Variable objects.
        """
        # get parameters of all layers and stretch them out into one list
        return [p for layer in self.layers for p in layer.parameters()]
    def save(self,file_path):
        """Save model parameters and state to disk.
        
        Parameters
        ----------
        file_path : str
            Path to save model.
        """
        self.state_dict = []
        for layer in self.layers:
            layer_state = {}
            if isinstance(layer, (Linear, BatchNorm)):
                layer_state['Params'] = [param.value for param in layer.parameters()]
                if isinstance(layer,BatchNorm):
                    layer_state['running_mean'] = layer.running_mean
                    layer_state['running_var'] = layer.running_var
            else:
                layer_state['Params'] = None
            self.state_dict.append(layer_state)
        with open(file_path, 'wb') as f:
            pickle.dump(self.state_dict, f)
        print(f"Model saved to {file_path}")    
                       
    def load(self,file_path):
        """Load model parameters and state from disk.
        
        Parameters
        ----------
        file_path : str
            Path to load model from.
        """
        with open(file_path, 'rb') as f:
            self.state_dict = pickle.load(f)
        for layer, layer_state in zip(self.layers,self.state_dict):
            if layer_state['Params'] is not None:
                for param,value in zip(layer.parameters(),layer_state['Params']):
                    if param.value.shape != value.shape:
                         print(f"Shape mismatch for layer: {layer}, expected {param.value.shape}, got {value.shape}")
                    param.value = value
                if isinstance(layer,BatchNorm):
                    layer.running_mean = layer_state.get('running_mean')
                    layer.running_var = layer_state.get('running_var')
        print(f"Model loaded from {file_path}")


#----------------------------------------------------------------------------------------------------------

class BatchNorm:
    """Batch Normalization layer.
    
    Normalizes activations across the batch dimension with learnable
    scale (gamma) and shift (beta) parameters. Maintains running statistics
    for inference.
    
    Parameters
    ----------
    dim : int
        Feature dimension to normalize.
    momentum : float, default=0.1
        Momentum for running statistics update.
    eps : float, default=1e-15
        Small constant for numerical stability.
        
    Attributes
    ----------
    gamma : Variable
        Learnable scale parameter of shape (1, dim).
    beta : Variable
        Learnable shift parameter of shape (1, dim).
    running_mean : ndarray
        Running mean for inference of shape (dim,).
    running_var : ndarray
        Running variance for inference of shape (dim,).
    training : bool
        Whether layer is in training mode.
    """
    def __init__(self,dim,momentum = 0.1,eps=1e-15) -> None:
        self.gamma = Variable(np.ones((1,dim)))
        self.beta = Variable(np.zeros((1,dim)))
        # training with momentum updata
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)  
        self.training = True 
        self.eps = eps
        self.momentum = momentum
        # intermediates for backward
        self.xvar = None
        self.xhat = None
        self.xnorm = None
    def __call__(self,x):
        """Forward pass with batch normalization.
        
        Parameters
        ----------
        x : ndarray, shape (batch_size, dim)
            Input activations.
            
        Returns
        -------
        out : ndarray, shape (batch_size, dim)
            Normalized, scaled, and shifted activations.
        """
        if self.training:
            xmean = np.mean(x,axis=0,keepdims=True)
            # m = x.shape[0]
            xvar = np.var(x,axis=0,keepdims=True) # unbiased estimate for the variance
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        else:
            #use running statistics
            xmean = self.running_mean
            xvar = self.running_var
        
        self.xhat = x - xmean
        self.xvar = (xvar + self.eps)
        # normalizing input
        self.xnorm = self.xhat/(self.xvar**0.5) # xhat ~N(0,1)
        # scaling and shifting
        self.out = self.gamma.value*self.xnorm + self.beta.value
        return self.out
    def backward(self,dal):
        """Backward pass for batch normalization.
        
        Parameters
        ----------
        dal : ndarray, shape (batch_size, dim)
            Gradient from subsequent layer.
            
        Returns
        -------
        dz : ndarray, shape (batch_size, dim)
            Gradient with respect to input.
        """
        batch_size = dal.shape[0]
        
        #grad for gamma and beta 
        self.gamma.grad = np.sum(dal * self.xnorm, axis=0)
        self.beta.grad = np.sum(dal, axis=0)
        
        # grad wrt xnorm
        dxnorm = dal * self.gamma.value
        
        #gradient wrt variancec(xvar)
        dvar = np.sum(dxnorm * self.xhat * -0.5 * (self.xvar ** -1.5), axis=0)
        
        # gradient wrt mean (xmean)
        dmean = np.sum(dxnorm * -1 / np.sqrt(self.xvar), axis=0) + dvar * np.mean(-2 * self.xhat, axis=0)
        # gradient wrt input x/z
        # dL/dzl = dL/dz_nl*dz_nl/dzl + dL/dvar* dvar/dzl + dL/dmue*dmue/dzl
        dz =  dxnorm/np.sqrt(self.xvar) + (2/batch_size)* dvar * self.xhat + dmean/batch_size
        return dz
    def parameters(self):
        """Return learnable parameters (gamma and beta)."""
        return [self.gamma,self.beta]

    """
    def backward01(self,dal):
        batch_size = dal.shape[0]
        self.dgama = np.sum(dal*self.xnorm,axis=0)
        self.dbeta = np.sum(dal,axis=0)
        # for dx we have take account of the mean and var we have calculated
        dz_ln = dal*self.gamma   # (batch_size,fan_out) *(1,fanout)  element-wise   broadcasting (1,fanout) -->(batch_size,fanout)
        # dvar = -0.5*np.ones(batch_size).T @ ( dz_ln * (self.xhat/self.xvar**(3/2)))     #  1.T @ ((batch_size,fanout) * (batch_size,fanout))  --> summing over the row (batch_size) resulting in (1,fanout)
        
        dvar = -0.5*np.sum( dz_ln * (self.xhat/self.xvar**(3/2)),axis=0)
        # for dmue there are two ways to achieve that  
        # dmue = (dL/dz_ln)*(dz_ln/dmue) + (dL/dvar)*(dvar/dmue) let say dmue1 and dmue2
        # dmue = -np.ones(batch_size).T @ (dz_ln * self.xvar**-0.5) # since dmue2 = 0 
        
        dmue = -np.sum(dz_ln / self.xvar**0.5,axis=0) + dvar*
        # from the DAg 
        # dL/dzl = dL/dz_nl*dz_nl/dzl + dL/dvar* dvar/dzl + dL/dmue*dmue/dzl
        
        # dz = (1/(batch_size* self.xvar**0.5))*(
        #     batch_size*dz_ln - np.ones((1,batch_size))@ dz_ln - self.xnorm*(np.ones((1,batch_size))@ (dz_ln * self.xnorm))
        # )
        dz = (1/(batch_size * self.xvar**0.5))*(
            batch_size*dz_ln - np.sum(dz_ln,axis=0) - self.xnorm*np.sum(dz_ln*self.xnorm,axis=0)
        )
        return dz
        
    """

#----------------------------------------------------------------------------------------------------------

class Dropout:
    """Dropout regularization layer.
    
    Randomly sets activations to zero during training to prevent
    overfitting. Scales remaining activations by 1/keep_prop.
    
    Parameters
    ----------
    keep_prop : float, default=0.9
        Probability of keeping each activation (between 0 and 1).
        
    Attributes
    ----------
    keep_prop : float
        Keep probability.
    training : bool
        Whether layer is in training mode.
    mask : ndarray or None
        Binary mask applied during training.
    """
    def __init__(self, keep_prop = 0.9):
        self.keep_prop = keep_prop
        self.training = True
        self.mask = None
    def __call__(self,x):
        """Forward pass with dropout.
        
        Parameters
        ----------
        x : ndarray
            Input activations.
            
        Returns
        -------
        out : ndarray
            Dropped and scaled activations (training) or unchanged input (eval).
        """
        if self.training:
            self.mask = (np.random.rand(*x.shape) < self.keep_prop).astype(np.float32)
            return x*self.mask/self.keep_prop
        else:
            # we don't want to use Dropout while evaluation
            return x
    def backward(self,dAl):
        """Backward pass for dropout.
        
        Parameters
        ----------
        dAl : ndarray
            Gradient from subsequent layer.
            
        Returns
        -------
        gradient : ndarray
            Masked gradient (training) or unchanged gradient (eval).
        """
        # gradient for the dropped neuron is zero while training 
        return dAl*self.mask if self.training else dAl
        
    def parameters(self):
        """Return trainable parameters (none for dropout)."""
        return []


#----------------------------------------------------------------------------------------------------------
