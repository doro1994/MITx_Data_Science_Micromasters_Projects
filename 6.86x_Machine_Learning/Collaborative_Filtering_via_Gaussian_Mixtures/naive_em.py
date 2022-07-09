"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

def m_norm(x, mu, var):
    return (np.exp(-0.5/var*np.dot(x - mu, x - mu)) / 
            (np.sqrt((2 * np.pi* var)**x.shape[0])))
    # return (np.exp(-0.5*np.linalg.norm(x - mu, ord = 1) / var) / 
    #         np.sqrt((2 * np.pi)**x.shape[0] * var))

from scipy.stats import multivariate_normal
var1 = multivariate_normal(mean=[3,4,5], cov=[[7,0,0],[0, 7, 0], [0, 0,7]])
var2 = np.random.multivariate_normal(mean=[3,4,5], cov=[[7,0,0],[0, 7, 0], [0, 0,7]])
print(var1.pdf([1,2,3]))
print(m_norm(np.array([1,2,3]), np.array([3,4,5]), 7))




def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n = len(X)
    K = mixture.p.shape[0]
    p_soft = np.zeros(n*K).reshape(n, K)
    likelihood = 0
    for i in range(len(X)):
        denominator = 0
        p = mixture.p
        mu = mixture.mu
        var = mixture.var
        for j in range(K):
            m_norm = (np.exp(-0.5 / var[j] * np.dot(X[i] - mu[j], X[i] - mu[j])) / 
                      np.sqrt((2 * np.pi * var[j])**X[i].shape[0]))
            if denominator == 0:
                for k in range(K):
                    m_norm_k = (np.exp(-0.5 / var[k] *np.dot(X[i] - mu[k], X[i] - mu[k])) /
                                np.sqrt((2 * np.pi * var[k])**X[i].shape[0]))
                    denominator += p[k] * m_norm_k
                p_soft[i, j] = p[j] * m_norm / denominator
                likelihood += np.log(denominator)
            else:
                p_soft[i, j] = p[j] * m_norm / denominator
    
    return (p_soft, likelihood)
            
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n = post.shape[0]
    K = post.shape[1]
    d = X.shape[1]
    entity_vector = np.ones(n) # (n,)
    weigted_inputs = np.matmul(post.transpose(), X) # (K, d)
    cluster_prob = np.matmul(post.transpose(), np.ones(n)) # (K,)
    mu = np.multiply(1/np.repeat(cluster_prob.reshape(K, 1), d, axis = 1), weigted_inputs) # (K, d)
    p = cluster_prob / n # (K,)
    var = np.zeros(K)
    #weighted_var = np.matmul(post.transpose(), X)
    for j in range(K):
        sub_mu = np.repeat(mu[j].reshape(1, d), n, axis = 0) # (n, d)
        #sub_var = np.matmul((X - sub_mu).transpose(), X - sub_mu) # (n, n)
        sub_var = np.linalg.norm(X - sub_mu, ord = 2, axis = 1) # (n,)
        weighted_var = np.dot(post.transpose()[j], sub_var**2) # (1,)
        var[j] = weighted_var / d / cluster_prob[j]
    return GaussianMixture(mu, var, p)
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    LL_old = -9999
    LL_new = -9998
    while LL_new - LL_old > (1e-6) * np.abs(LL_new):
        LL_old = LL_new
        p_soft, LL_new = estep(X, mixture)
        mixture = mstep(X, p_soft)
        #print("LL:", LL_new)
    return (mixture, p_soft, LL_new)
    raise NotImplementedError
