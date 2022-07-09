# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:55:02 2020

@author: user
"""
import numpy as np
import math
from collections import defaultdict


X = np.array([[0, -6],
              [4, 4],
              [0, 0],
              [-5, 2]])
k = 2



def k_means(X, dist, mu1, mu2, **kwargs):
    cluster_1 = []
    cluster_2 = []
    mu1_prev = np.array([0, 0])
    mu2_prev = np.array([0, 0])
    while not ((mu1 == mu1_prev).all() and (mu2 == mu2_prev).all()):
        for x in X:
            if dist(x - mu1, **kwargs) <= dist(x - mu2, **kwargs):
                if tuple(x) not in cluster_1: 
                    cluster_1.append(tuple(x))
                    try: 
                        cluster_2.remove(tuple(x))
                    except:
                        pass
            else:
                if tuple(x) not in cluster_2:
                    cluster_2.append(tuple(x))
                    try:
                        cluster_1.remove(tuple(x))
                    except:
                        pass
        mu1_prev = mu1
        mu2_prev = mu2
        mu1 = np.mean(cluster_1, axis = 0)
        mu2 = np.mean(cluster_2, axis = 0)
    
    return (mu1, cluster_1, mu2, cluster_2)

def k_medoids(X, dist, mu1, mu2, **kwargs):
    cluster_1 = []
    cluster_2 = []
    mu1_prev = np.array([0, 0])
    mu2_prev = np.array([0, 0])
    total_cost = 999
    while not ((mu1 == mu1_prev).all() and (mu2 == mu2_prev).all()):
        for x in X:
            if dist(x - mu1, **kwargs) <= dist(x - mu2, **kwargs):
                if tuple(x) not in cluster_1: 
                    cluster_1.append(tuple(x))
                    try: 
                        cluster_2.remove(tuple(x))
                    except:
                        pass
            else:
                if tuple(x) not in cluster_2:
                    cluster_2.append(tuple(x))
                    try:
                        cluster_1.remove(tuple(x))
                    except:
                        pass
        mu1_prev = mu1
        mu2_prev = mu2
        
        
        medoids = [mu1, mu2]
        new_medoids = [mu1, mu2]
        clusters = [cluster_1, cluster_2]
 
        
        mu1_mean = np.mean(cluster_1, axis = 0)
        mu2_mean = np.mean(cluster_2, axis = 0)
        means = [mu1_mean, mu2_mean]
        minimal_dist = 999
        
        for medoid_id in range(len(medoids)):
            for x in X:
                if tuple(x) in clusters[medoid_id]:
                    if dist(x - means[medoid_id], **kwargs) < minimal_dist:
                        minimal_dist = dist(x - means[medoid_id], **kwargs)
                        new_medoids[medoid_id] = x
        mu1 = new_medoids[0]
        mu2 = new_medoids[1]                    
        
        # for medoid_id in range(len(medoids)):
        #     cost = 999.0
        #     best_cost_change = 0
        #     for x in X:
        #         if ((not (medoids[medoid_id] == x).all())
        #             and (tuple(x) in clusters[medoid_id])):
        #             current_cost = dist(x - medoids[medoid_id], **kwargs)
        #             if current_cost < cost:
        #                 cost = current_cost
        #                 minimal_cost = cost
        #                 new_medoids[medoid_id] = x
        # mu1 = new_medoids[0]
        # mu2 = new_medoids[1]

    
    return (mu1, cluster_1, mu2, cluster_2)

# mu1, cluster_1, mu2, cluster_2 = k_medoids(X, np.linalg.norm, X[0], X[-1], ord = 2)
# print("Cluster 1 Center:", mu1)
# print("Cluster 1 Members:", cluster_1)
# print("Cluster 2 Center:", mu2)
# print("Cluster 2 Members:", cluster_2)

theta_0 = np.array([0.5, 0.5, 6, 7, 1, 4])
pi_1 = 0.5
pi_2 = 0.5
pi = [0.5, 0.5]
mu_1 = 6
mu_2 = 7
mu = [mu_1, mu_2]
var_1 = 1
var_2 = 4
var = [var_1, var_2]
X = np.array([-1, 0, 4, 5, 6])
LL = -999
LL_prev = -1000
def pnorm(x, mu, var):
    return np.sqrt(1/(2*np.pi*var))*np.exp(-0.5*(x - mu)**2 / var)

while (LL - LL_prev) > 0.1:
    LL_prev = LL
    p = defaultdict(dict)
    # E-step
    for k in range(2):
        for x in X:
            p[k][x] = pi[k] * pnorm(x, mu[k], var[k]) / (pi[0] * pnorm(x, mu[0], var[0]) + pi[1] * pnorm(x, mu[1], var[1]))
    
    LL = 0
    # M-step
    # for k in range(2):
    #     for x in X:
    #         LL += np.log(pi[k] * pnorm(x, mu[k], var[k]))
    # LL = 0       
    for x in X:
        LL += np.log(pi_1 * pnorm(x, mu_1, var_1) + pi_2 * pnorm(x, mu_2, var_2))
    # print(LL)
            
    # for x in X:
    #     if p[1][x] > p[0][x]:
    #         print("Point", x, ": 2")
    #     else:
    #         print("Point", x, ": 1")
            
    mu_1_updated = sum([x*p[0][x] for x in X]) / sum([p[0][x] for x in X])
    mu_2_updated = sum([x*p[1][x] for x in X]) / sum([p[1][x] for x in X])
    var_1_updated = sum([p[0][x] * (x - mu_1_updated)**2 for x in X]) / sum([p[0][x] for x in X])
    var_2_updated = sum([p[1][x] * (x - mu_2_updated)**2 for x in X]) / sum([p[1][x] for x in X])
    mu_1 = mu_1_updated
    mu_2 = mu_2_updated
    var_1 = var_1_updated
    var_2 = var_2_updated
    mu = [mu_1, mu_2]
    var = [var_1, var_2]
print(var)
    


