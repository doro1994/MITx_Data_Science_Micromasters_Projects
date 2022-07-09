
import numpy as np

# 1. Collaborative Filtering, Kernels, Linear Regression

U_0 = np.array([[6, 0, 3, 6]])
V_0 = np.array([[4, 2, 1]])

X_0 = np.matmul(U_0.transpose(), V_0)
Y = np.array

SE = 0.5 * ((5-24)**2 + (7-6)**2 + (2-0)**2 + (4-12)**2 + (12-3)**2 + (6-6)**2)
Reg = 0.5 * (np.linalg.norm(U_0)**2 + np.linalg.norm(V_0)**2)

U_1 = np.array([[3, 2, 4/3, -3/2]])

X_1 = np.matmul(U_1.transpose(), V_0)

A = np.array([[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
              [1/3, 1/3, 1/3,-1/3,-1/3,-1/3]])


print(np.matmul(A,A.transpose()))
print(np.linalg.inv(A))