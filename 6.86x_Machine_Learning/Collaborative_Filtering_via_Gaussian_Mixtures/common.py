"""Mixture model for collaborative filtering"""
from typing import NamedTuple, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc


class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component


def init(X: np.ndarray, K: int,
         seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """Initializes the mixture model with random points as initial
    means and uniform assingments

    Args:
        X: (n, d) array holding the data
        K: number of components
        seed: random seed

    Returns:
        mixture: the initialized gaussian mixture
        post: (n, K) array holding the soft counts
            for all components for all examples

    """
    np.random.seed(seed)
    n, _ = X.shape
    p = np.ones(K) / K

    # select K random points as initial means
    mu = X[np.random.choice(n, K, replace=False)]
    var = np.zeros(K)
    # Compute variance
    for j in range(K):
        var[j] = ((X - mu[j])**2).mean()

    mixture = GaussianMixture(mu, var, p)
    post = np.ones((n, K)) / K

    return mixture, post


def plot(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray,
         title: str):
    """Plots the mixture model for 2D data"""
    _, K = post.shape

    percent = post / post.sum(axis=1).reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    r = 0.25
    color = ["r", "b", "k", "y", "m", "c"]
    for i, point in enumerate(X):
        theta = 0
        for j in range(K):
            offset = percent[i, j] * 360
            arc = Arc(point,
                      r,
                      r,
                      0,
                      theta,
                      theta + offset,
                      edgecolor=color[j])
            ax.add_patch(arc)
            theta += offset
    for j in range(K):
        mu = mixture.mu[j]
        sigma = np.sqrt(mixture.var[j])
        circle = Circle(mu, sigma, color=color[j], fill=False)
        ax.add_patch(circle)
        legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(
            mu[0], mu[1], sigma)
        ax.text(mu[0], mu[1], legend)
    plt.axis('equal')
    plt.show()


def rmse(X, Y):
    return np.sqrt(np.mean((X - Y)**2))

def bic(X: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians

    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data

    Returns:
        float: the BIC for this mixture
    """
    p = mixture.p # (K,)
    mu = mixture.mu
    var = mixture.var
    #K = var.shape[0]
    n = X.shape[0]
    #d = X.shape[1]
    parameters = p.shape[0] + mu.shape[0]*mu.shape[1] + var.shape[0] - 1 
    # distance = np.linalg.norm(X - mu[:, None, :]) # (K, n)
    # variance = var.reshape(1, K)
    # N_matrix = (1 / (2 * np.pi * variance)**(d / 2) * 
    #             np.exp(-distance**2 / (2 * variance))) # (K, n)
    # LL = np.sum(np.log(np.sum(p * N_matrix, axis = 0)), axis = 0)
    return log_likelihood - 0.5 * parameters * np.log(n)
    
    
    raise NotImplementedError

X =np.array([[0.85794562, 0.84725174],
 [0.6235637,  0.38438171],
 [0.29753461, 0.05671298],
 [0.27265629, 0.47766512],
 [0.81216873, 0.47997717],
 [0.3927848,  0.83607876],
 [0.33739616, 0.64817187],
 [0.36824154, 0.95715516],
 [0.14035078, 0.87008726],
 [0.47360805, 0.80091075],
 [0.52047748, 0.67887953],
 [0.72063265, 0.58201979],
 [0.53737323, 0.75861562],
 [0.10590761, 0.47360042],
 [0.18633234, 0.73691818]])
K = 6
Mu = np.array([[0.6235637,  0.38438171],
 [0.3927848,  0.83607876],
 [0.81216873, 0.47997717],
 [0.14035078, 0.87008726],
 [0.36824154, 0.95715516],
 [0.10590761, 0.47360042]])
Var = np.array([0.10038354, 0.07227467, 0.13240693, 0.12411825, 0.10497521, 0.12220856])
P = np.array([0.1680912,  0.15835331, 0.21384187, 0.14223565, 0.14295074, 0.17452722])
LL = -1424.283792

mixt = GaussianMixture(Mu, Var, P)
print(bic(X, mixt, LL))
# bix:-1447.302219
# bix)correct:-1455.426369