
import random
import numpy as np

normal = random.choice([1, 2])

X_list = []
Y_list = []
XY_list = []

for _ in range(100000):
    normal = random.choice([1, 2])
    if normal == 1:
        w = 1
        mu = 2
    else:
        w = -1
        mu = -2
        
    p = 0.5
    var = 1
    
    X = np.random.normal(mu, var, 1)
    eps = np.random.normal(0, 1, 1)
    Y = w*X[0] + eps[0]
    
    
    X_list.append(X[0])
    Y_list.append(Y)
    XY_list.append(X[0]*Y)

print("EX=", np.mean(X_list))
print("EY=", np.mean(Y_list))
print("EXY=", np.mean(XY_list))