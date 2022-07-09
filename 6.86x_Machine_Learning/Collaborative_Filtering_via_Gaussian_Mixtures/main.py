import numpy as np
import kmeans
import common
import naive_em
import em

#X = np.loadtxt("toy_data.txt")
X = np.loadtxt("netflix_incomplete.txt")

"""
for K in [1, 2, 3, 4]:
    cost_min = 10**6
    title = "K = " + str(K) 
    for seed in [0, 1, 2, 3, 4]:  
        mixture, post = common.init(X, K, seed)        
        mixture, post, cost = kmeans.run(X, mixture, post)
        if cost < cost_min:
            cost_min = cost
            mixture_best = mixture
            post_best = post
    common.plot(X, mixture_best, post_best, title)
    print("Cost for K = ", K, ":", cost_min)

"""


# X = np.array([[0.85794562, 0.84725174],
#               [0.6235637,  0.38438171],
#               [0.29753461, 0.05671298],
#               [0.27265629, 0.47766512],
#               [0.81216873, 0.47997717],
#               [0.3927848,  0.83607876],
#               [0.33739616, 0.64817187],
#               [0.36824154, 0.95715516],
#               [0.14035078, 0.87008726],
#               [0.47360805, 0.80091075],
#               [0.52047748, 0.67887953],
#               [0.72063265, 0.58201979],
#               [0.53737323, 0.75861562],
#               [0.10590761, 0.47360042],
#               [0.18633234, 0.73691818]])

# # test
# X = np.array(
# [[2., 5., 3., 0., 0.],
#   [3., 5., 0., 4., 3.],
#   [2., 0., 3., 3., 1.],
#   [4., 0., 4., 5., 2.],
#   [3., 4., 0., 0., 4.],
#   [1., 0., 4., 5., 5.],
#   [2., 5., 0., 0., 1.],
#   [3., 0., 5., 4., 3.],
#   [0., 5., 3., 3., 3.],
#   [2., 0., 0., 3., 3.],
#   [3., 4., 3., 3., 3.],
#   [1., 5., 3., 0., 1.],
#   [4., 5., 3., 4., 3.],
#   [1., 4., 0., 5., 2.],
#   [1., 5., 3., 3., 5.],
#   [3., 5., 3., 4., 3.],
#   [3., 0., 0., 4., 2.],
#   [3., 5., 3., 5., 1.],
#   [2., 4., 5., 5., 0.],
#   [2., 5., 4., 4., 2.]])
# K = 4
# Mu = np.array(
# [[2., 4., 5., 5., 0.],
#   [3., 5., 0., 4., 3.],
#   [2., 5., 4., 4., 2.],
#   [0., 5., 3., 3., 3.]])
# Var = np.array([5.93, 4.87, 3.99, 4.51])
# P = np.array([0.25, 0.25, 0.25, 0.25])


# mixture = common.GaussianMixture(Mu, Var, P)
# _, post = common.init(X, K, 0) 


# mixture, p_soft, LL = em.run(X, mixture, post)

# X_pred = em.fill_matrix(X, mixture)


# X_gold = np.array(
# [[2., 5., 3., 4., 3.],
#  [3., 5., 3., 4., 3.],
#  [2., 4., 3., 3., 1.],
#  [4., 4., 4., 5., 2.],
#  [3., 4., 4., 4., 4.],
#  [1., 5., 4., 5., 5.],
#  [2., 5., 4., 5., 1.],
#  [3., 4., 5., 4., 3.],
#  [3., 5., 3., 3., 3.],
#  [2., 5., 3., 3., 3.],
#  [3., 4., 3., 3., 3.],
#  [1., 5., 3., 5., 1.],
#  [4., 5., 3., 4., 3.],
#  [1., 4., 3., 5., 2.],
#  [1., 5., 3., 3., 5.],
#  [3., 5., 3., 4., 3.],
#  [3., 5., 4., 4., 2.],
#  [3., 5., 3., 5., 1.],
#  [2., 4., 5., 5., 3.],
#  [2., 5., 4., 4., 2.]])

# print(common.rmse(X_gold, X_pred))   

# p_soft, LL = em.estep(X, mixture)
# print("post:", p_soft)
# print("LL:", LL)


# mixture = em.mstep(X, p_soft, mixture)

# print("Mu:", mixture.mu)
# print("Var:", mixture.var)
# print("P:", mixture.p)



BIC_best = -10**6 
for K in [1, 12]:
    LL_max = -10**20
    title = "K = " + str(K) 
    for seed in [0, 1, 2, 3, 4]:  

        # mu = np.array([[0.6235637,  0.38438171],
        #                 [0.3927848,  0.83607876],
        #                 [0.81216873, 0.47997717],
        #                 [0.14035078, 0.87008726],
        #                 [0.36824154, 0.95715516],
        #                 [0.10590761, 0.47360042]])
        # var = np.array([0.10038354,
        #                 0.07227467,
        #                 0.13240693,
        #                 0.12411825,
        #                 0.10497521,
        #                 0.12220856])
        # p = np.array([0.1680912,
        #             0.15835331,
        #             0.21384187,
        #             0.14223565,
        #             0.14295074,
        #             0.17452722])
        mixture, post = common.init(X, K, seed)  
        # mixture = common.GaussianMixture(Mu, Var, P)
        # p_soft, LL = em.estep(X, mixture)
        # mixture = em.mstep(X, p_soft, mixture)
        #print(LL)
        mixture, p_soft, LL = em.run(X, mixture, post)
        print(LL)
        if LL > LL_max:
            mixture_best = mixture
            p_soft_best = p_soft
            LL_max = LL
    print(title, ", LL =", LL_max)
    # BIC = common.bic(X, mixture_best, LL_max)
X_pred = em.fill_matrix(X, mixture_best)
X_gold = np.loadtxt('netflix_complete.txt')



print(common.rmse(X_gold, X_pred))   
    # if BIC > BIC_best:
    #     BIC_best = BIC
    #     K_best = K
    #print(title, ", LL =", LL)


#    # common.plot(X, mixture_best, p_soft_best, title)
# print("BIC_best:", BIC_best)
# print("K_best:", K_best)
