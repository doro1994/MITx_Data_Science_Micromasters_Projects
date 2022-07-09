import numpy as np

X = np.array([[1, 0, 1], 
             [1, 1, 1],
             [1, 1,-1],
             [-1,1, 1]])
y = np.array([2, 2.7, -0.7, 2])

theta = np.array([0, 1, 2])

def Hinge_Loss(z):
    if z >= 1:
        return 0
    else:
        return 1 - z
    
def Sqared_error_loss(z):
    return (z**2)/2

R = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    R[i] = Hinge_Loss(y[i] - np.matmul(theta, X[i]))
print(R.mean())

R = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    R[i] = Sqared_error_loss(y[i] - np.matmul(theta, X[i]))
print(R.mean())