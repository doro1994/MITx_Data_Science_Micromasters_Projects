"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    delta = (X > 0).astype(int)
    n = len(X)
    K = mixture.p.shape[0]
    p_soft = np.zeros(n*K).reshape(n, K)
    f = np.zeros(n*K).reshape(n, K)
    likelihood = 0
    for i in range(len(X)):
        denominator_log = 0
        p = mixture.p
        mu = mixture.mu
        var = mixture.var
        Cu = np.nonzero((X[i] > 0).astype(int))
        X_Cu = X[i]
        X_Cu = X_Cu[Cu]
        for j in range(K):
            mu_Cu = mu[j]
            mu_Cu = mu_Cu[Cu]
            # m_norm = (np.exp(-0.5 / var[j] * np.dot(X_Cu - mu_Cu, X_Cu - mu_Cu)) / 
            #           np.sqrt((2 * np.pi * var[j])**X_Cu.shape[0]))
            # print(m_norm)
            # f[i, j] = np.log(p[j] + 1e-16) + np.log(m_norm)
            f[i, j] = (np.log(p[j] + 1e-16) - 0.5 / var[j] * np.dot(X_Cu - mu_Cu, X_Cu - mu_Cu) -
                    X_Cu.shape[0] / 2 * np.log((2 * np.pi * var[j])))
    max_f = np.max(f)
    L = f - max_f - logsumexp(f - max_f, axis = 1).reshape(n, 1)
    likelihood = np.sum(max_f + logsumexp(f - max_f, axis = 1), axis = 0)
    
    return (np.exp(L), likelihood)  
    
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    delta = (X > 0).astype(int) # (n, d)
    n = len(X)
    K = mixture.p.shape[0]
    d = X.shape[1]
    # for j in range(K):
    #     p_ju = post[:, j] # (n,)
    # if np.matmul(post[:, j].transpose(), delta) >= 1
    #     mu = np.matmul(post.transpose(), delta * X) / np.matmul(post.transpose(), delta) # (K,d)
    # else:
    #     mu = mixture.mu
        
    mu = np.matmul(post.transpose(), delta * X) / np.matmul(post.transpose(), delta) # (K,d)
    no_update = np.matmul(post.transpose(), delta) < 1 # (K, d)
    mu[no_update] = mixture.mu[no_update]
    mu = mu.reshape(K, d)   
    p = (1/n) * np.sum(post, axis = 0)
    var = np.zeros(K)
    distance = np.linalg.norm(X - mu[:, None, :], ord = 2, axis = 2) # (K, n)
    numerator = 0
    denumerator = 0
    for i in range(n):
        X_Cu = X[i] # (1, d)
        delta_X = (X_Cu > 0).astype(int) # (1, d)
        numerator += post[i, :] * (np.linalg.norm(X_Cu - mu * delta_X, ord = 2, axis = 1)**2)  #(K,)
        denumerator += np.sum(delta_X) * post[i, :]
    var = numerator / np.sum(np.matmul(post.transpose(), np.sum(delta, axis = 1).reshape(n, 1)), axis = 1)
    var = numerator / denumerator        
    var[var < min_variance] = min_variance
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
    LL_old = -10**10
    LL_new = -10**9
    count = 0
    while LL_new - LL_old > (1e-6) * np.abs(LL_new):
        LL_old = LL_new
        p_soft, LL_new = estep(X, mixture)
        mixture = mstep(X, p_soft, mixture)
        count += 1
        #print("LL:", LL_new)
    print("count", count)
    return (mixture, p_soft, LL_new)
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    p = mixture.p
    mu = mixture.mu
    n = X.shape[0]
    K = p.shape[0]
    X_pred = np.copy(X)
                
    delta = (X > 0).astype(int)
    n = len(X)
    K = mixture.p.shape[0]
    p_soft = np.zeros(n*K).reshape(n, K)
    f = np.zeros(n*K).reshape(n, K)
    likelihood = 0
    for i in range(len(X)):
        denominator_log = 0
        p = mixture.p
        mu = mixture.mu
        var = mixture.var
        Cu = np.nonzero((X[i] > 0).astype(int))
        X_Cu = X[i]
        X_Cu = X_Cu[Cu]
        for j in range(K):
            mu_Cu = mu[j]
            mu_Cu = mu_Cu[Cu]
            f[i, j] = (np.log(p[j] + 1e-16) - 0.5 / var[j] * np.dot(X_Cu - mu_Cu, X_Cu - mu_Cu) -
                    X_Cu.shape[0] / 2 * np.log((2 * np.pi * var[j])))
    max_f = np.max(f)
    L = f - max_f - logsumexp(f - max_f, axis = 1).reshape(n, 1)
    likelihood = np.sum(max_f + logsumexp(f - max_f, axis = 1), axis = 0)    
    post = np.exp(L) # (n, K)
    to_fill = X_pred == 0
    X_pred[to_fill] = np.dot(post, mu)[to_fill]
    
    return X_pred
    raise NotImplementedError
