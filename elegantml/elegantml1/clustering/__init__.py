"""
Clustering Module
=================

This module provides clustering algorithms.

Classes:
--------
- KMeans: K-Means clustering algorithm
- DBSCAN: Density-based spatial clustering (TODO)
"""

import numpy as np

__all__ = ['KMeans']


class KMeans:
    """
    K-Means Clustering Algorithm.
    
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form
    max_iter : int, default=300
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, default=None
        Random seed for reproducibility
        
    Examples
    --------
    >>> from elegantML.clustering import KMeans
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0)
    >>> kmeans.fit(X)
    >>> kmeans.labels_
    array([0, 0, 0, 1, 1, 1])
    """
    
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        
    def _initialize_centroids(self, X):
        """Initialize centroids randomly from data points."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]
    
    def _assign_clusters(self, X):
        """Assign each sample to the nearest centroid."""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, labels):
        """Update centroids as the mean of assigned samples."""
        new_centroids = np.array([X[labels == k].mean(axis=0) 
                                  for k in range(self.n_clusters)])
        return new_centroids
    
    def _compute_inertia(self, X, labels):
        """Compute sum of squared distances to nearest centroid."""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k])**2)
        return inertia
    
    def fit(self, X):
        """
        Compute k-means clustering.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training instances
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # Assign samples to nearest centroid
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
                
            self.centroids = new_centroids
        
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels)
        
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample."""
        self.fit(X)
        return self.labels_
