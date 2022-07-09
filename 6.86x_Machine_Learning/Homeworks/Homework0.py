# 2. (a)
import numpy as np

X = np.array([[-4, 2],
             [-2, 1],
             [-1,-1],
             [ 2, 2],
             [ 1,-2]]*10)

y = np.array([1, 1, -1, -1, -1]*10)

theta = np.array([-3, 3])
theta_0 = -3
for i in range(0, 50):
    if y[i] * (np.matmul(theta.transpose(), X[i]) + theta_0) <= 0:
        theta = theta + y[i]*X[i]
        theta_0 = theta_0 + y[i]
        print(X[i], "is missclassified")
print(theta, theta_0)

# 3. (a)
def f(x1, x2, x3):
    return ((1-x1)*(1-x2)*(1-x3))
f(0, 0, 0)
f(1, 1, 1)

theta = np.array([1, 1])
x = np.array([-2, 1])
y = 1
lambd = 3
eta = 1/lambd
theta_1_Hinge = theta + eta*y*x
theta_1_ZeroOne = theta + y*x
print(theta_1_Hinge)