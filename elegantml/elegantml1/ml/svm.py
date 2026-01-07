"""
Support Vector Machines
======================

Minimal, NumPy-first implementations for Support Vector Classification (SVC)
and Support Vector Regression (SVR) with selectable kernels.

Methods
-------
- `fit(X, y)`: train SVM via SMO-style updates (SVC) or dual optimization (SVR)
- `predict(X)`: predictions for inputs (SVC: labels; SVR: values)
- `get_support_vectors()`: access learned support vectors
"""

import numpy as np
from .base import BaseModel

__all__ = ["SupportVectorClassifier", "SupportVectorRegressor"]


class SupportVectorClassifier(BaseModel):
    """Support Vector Classifier (SVC).

    Parameters
    ----------
    C : float, default 1.0
        Regularization parameter.
    kernel_type : {"linear", "quadratic", "rbf"}, default "linear"
        Kernel to use.
    lr : float, default 0.01
        Learning rate (if applicable).
    max_iter : int, default 1000
        Maximum iterations.
    eps : float, default 1e-5
        Convergence tolerance and support vector threshold.

    Attributes
    ----------
    alpha : ndarray or None
        Dual variables.
    support_vectors_ : ndarray or None
        Support vectors.
    b : float
        Bias term.
    X_train, Y_train : ndarray or None
        Cached training data.
    """

    def __init__(self, C: float = 1.0, kernel_type: str = "linear", lr: float = 0.01, max_iter: int = 1000, eps: float = 1e-5) -> None:
        self.C = C
        self.kernels = {"linear": self._linear_kernel,
                        "quadratic": self._quadratic_kernel,
                          "rbf": self._rbf_kernel}
        self.kernel = self.kernels.get(kernel_type, self._linear_kernel)
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps
        self.alpha: np.ndarray | None = None
        self.support_vectors_: np.ndarray | None = None
        self.b: float = 0.0
        self.X_train: np.ndarray | None = None
        self.Y_train: np.ndarray | None = None
    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """ 
        Linear kernel: K(x,y) = x.y
        x1: shape (n_samples_1, n_features)
        x2: shape (n_samples_2, n_features)
        return : shape (n_samples_1, n_samples_2)
        """
        return np.dot(X1, X2.T)
    def _quadratic_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """  Quadratic kernel: K(x,y) = (x.y + 1)^2 
        x1 : shape (n_samples_1, n_features)
        x2 : shape (n_samples_2, n_features)
        return : shape (n_samples_1, n_samples_2)
        """
        return (np.dot(X1, X2.T) + 1) ** 2
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray, gamma: float = 0.1) -> np.ndarray:
        """ RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2) 
        x1 : shape (n_samples_1, n_features)
        x2 : shape (n_samples_2, n_features)
        return : shape (n_samples_1, n_samples_2)       
        """
        diff = X1 - X2
        return np.exp(-gamma *np.dot(diff, diff.T))
    
    def calc_b(self):
        """ Calculate bias term b. """
        y_support = self.Y_train[self.alpha > self.eps]
        k_support = self.kernel(self.X_train[self.alpha > self.eps], self.X_train)
        b = np.mean(y_support - np.sum(self.alpha* self.Y_train * k_support, axis = 0))
        return b
        
    def hypothesis(self,X: np.ndarray) -> np.ndarray:
        """Decision function for SVC.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : ndarray, shape (n_samples,)
            Signed distances from the decision boundary.
        """
        k_test = self.kernel(X,self.X_train)
        return np.sum(self.alpha * self.Y_train * k_test, axis = 1) + self.b
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupportVectorClassifier":
        """Fit the SVC using an SMO-style optimization.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training inputs.
        y : ndarray, shape (n_samples,)
            Binary labels in {+1, -1}.

        Returns
        -------
        self : SupportVectorClassifier
            Fitted estimator.
        """ 
        self.X_train = X
        self.Y_train = y
        n,d = X.shape
        self._K = self.kernel(X, X)
        self.alpha = np.zeros(n)
        for iter in range(self.max_iter):
            alpha_prev = self.alpha.copy()
            for j in range(0,n):
                #Generate random integer in [0,n] excluding 'j'. 
                i = j
                while i == j:
                    i = np.random.randint(0, n)
                kij = self._K[i,i] + self._K[j,j] - 2*self._K[i,j]
                if kij == 0:
                    continue
                alphaj_old,alphai_old = self.alpha[j], self.alpha[i]
                if y[i] != y[j]:
                    L = max(0, alphaj_old - alphai_old)
                    H = min(self.C, self.C + alphaj_old - alphai_old) # is it correct?
                else:
                    L = max(0, alphai_old + alphaj_old - self.C)
                    H = min(self.C, alphai_old + alphaj_old)
                # L,H = self._compute_L_H(self.C,alphai_old,alphaj_old,y[i],y[j])
                self.b = self.calc_b()
                # calculate E_i, E_j
                E_i = self.hypothesis(X[i]) - y[i]
                E_j = self.hypothesis(X[j]) - y[j]
                
                self.alpha[j] = alphaj_old + (y[j]*(E_i - E_j)) / kij
                
                # alpha[j] clipped to [L,H]
                self.alpha[j] = min(H, max(L, self.alpha[j]))
                
                #update alphai
                self.alpha[i] = alphai_old + y[i]*y[j]*(alphaj_old - self.alpha[j])
            # check convergence
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.eps:
                break  
        self.b = self.calc_b()     
        return self
    def get_support_vectors(self) -> np.ndarray:
        """Return support vectors for the trained SVC.

        Returns
        -------
        SV : ndarray, shape (n_support, n_features)
            Support vectors.
        """
        if self.alpha is None or self.X_train is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        idx = np.where(self.alpha > self.eps)[0]
        return self.X_train[idx]
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels using the decision function sign.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted labels in {+1, -1}.
        """
        if self.w is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        scores = self.hypothesis(X.T)
        return np.sign(scores)


class SupportVectorRegressor(BaseModel):
    """Support Vector Regression (SVR).

    Parameters
    ----------
    C : float, default 1.0
        Regularization parameter.
    epsilon : float, default 0.1
        Epsilon-insensitive tube width.
    kernel_type : {"linear", "quadratic", "rbf"}, default "linear"
        Kernel to use.
    max_iter : int, default 1000
        Maximum iterations.
    tol : float, default 1e-4
        Numerical tolerance.

    Attributes
    ----------
    alpha, alpha_star : ndarray or None
        Dual variables.
    b : float
        Bias term.
    X_train, Y_train : ndarray or None
        Cached training data.
    """

    def __init__(self,C : float = 1.0, epsilon: float = 0.1,kernel_type: str = "linear", max_iter: int = 1000,tol: float = 1e-4) -> None:
        self.C = C
        self.epsilon = epsilon
        self.kernel_type = kernel_type
        self.max_iter = max_iter
        self.tol = tol
        self.b: float = 0.0
        self.kernels = {"linear": self._linear_kernel,
                        "quadratic": self._quadratic_kernel,
                            "rbf": self._rbf_kernel}
        self.kernel = self.kernels.get(kernel_type, self._linear_kernel)
        
        #model parameters
        self.alpha: np.ndarray | None = None
        self.alpha_star: np.ndarray | None = None
        self._K: np.ndarray | None = None
        
        self.X_train: np.ndarray | None = None
        self.Y_train: np.ndarray | None = None
        
        
        
    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.dot(X1, X2.T)
    def _quadratic_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return (np.dot(X1, X2.T) + 1) ** 2
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray, gamma: float = 0.1) -> np.ndarray:
        """ RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2) """
        # Should use squared Euclidean distance matrix
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sq_dists)
    def hypothesis(self,X : np.ndarray)-> np.ndarray:
        """Decision function for SVR.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Predicted values.
        """
        ktest = self.kernel(X,self.X_train)
        return np.sum((self.alpha_star - self.alpha)* self.Y_train * ktest, axis = 1) + self.b
    def calc_b(self):
        """Compute the SVR bias term `b` using boundary support vectors."""
        bndr_sv_mask  = ((self.alpha > self.tol) & (self.alpha < self.C - self.tol) |
                         (self.alpha_star > self.tol) & (self.alpha_star < self.C - self.tol))
        X_bndr = self.X_train[bndr_sv_mask]
        y_bndr = self.Y_train[bndr_sv_mask]
        k_support = self.kernel(X_bndr,self.X_train)
        KC = np.sum((self.alpha_star - self.alpha) * k_support, axis = 0)
        b_vals = y_bndr - KC
        b_vals -= self.epsilon *((self.alpha[bndr_sv_mask] > self.tol) & (self.alpha < self.C - self.tol)).astype(float)
        b_vals += self.epsilon *((self.alpha_star[bndr_sv_mask] > self.tol) & (self.alpha_star < self.C - self.tol)).astype(float)
        b = np.mean(b_vals) 
        return b
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupportVectorRegressor":
        """Fit the SVR model via dual optimization.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training inputs.
        y : ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        self : SupportVectorRegressor
            Fitted estimator.
        """
        n,d = X.shape
        # compute Kernel matrix
        self._K = self._linear_kernel(X, X)
        # dual variables 
        beta = np.zeros(2*n)
        sign = np.concatenate([-1*np.ones(n), np.ones(n)])
        tau = np.concatenate([-y,y])
        #! do we really need the full 2n x 2n kernel matrix?
        Q = np.block([[self._K, -self._K],
                               [-self._K, self._K]]) #shape (2n,2n)
        
        for iter in range(self.max_iter):
            beta_prev = beta.copy()
            indices = np.arange(2*n)
            np.random.shuffle(indices)
            for j in indices:
                #Generate random integer in [0,2n] excluding 'j'. 
                i = j
                while i == j:
                    i = np.random.randint(0, 2*n)
                eta = Q[j,j] + Q[i,i] - 2*Q[i,j]
                if eta == 0:
                    continue
                beta_j_old, beta_i_old = beta[j], beta[i]
                # compute L and H
                if sign[i] != sign[j]:
                    L = max(0, beta_j_old - beta_i_old)
                    H = min(self.C, self.C + beta_j_old - beta_i_old)
                else:
                    L = max(0, beta_i_old + beta_j_old - self.C)
                    H = min(self.C, beta_i_old + beta_j_old)
                # compute E_i, E_j
                E_i = np.sum(beta * sign * Q[:,i]) - tau[i] - self.epsilon*sign[i]
                E_j = np.sum(beta * sign * Q[:,j]) - tau[j] - self.epsilon*sign[j]
                # update beta_j
                beta[j] = beta_j_old + (sign[j]*(E_i - E_j)) / eta
                # clip beta_j to [L,H]
                beta[j] = min(H, max(L, beta[j]))
                # update beta_i
                beta[i] = beta_i_old + sign[i]*sign[j]*(beta_j_old - beta[j])
            # check convergence
            diff = np.linalg.norm(beta - beta_prev)
            if diff < 1e-5:
                break
        self.alpha = beta[:n]
        self.alpha_star = beta[n:]
        self.sv_mask = (self.alpha > self.eps) | (self.alpha_star > self.eps)
        self.b = self.calc_b()
        return self
        #update b here if needed
    def get_support_vectors(self) -> np.ndarray:
        if self.alpha is None or self.alpha_star is None or self.X_train is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        return self.X_train[self.sv_mask]
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for inputs using the SVR model.
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input samples.
        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted target values.
        """
        return self.hypothesis(X)