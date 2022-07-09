import numpy as np

### Functions for you to fill in ###

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    n = X.shape[1]
    theta = np.matmul(np.linalg.inv(np.matmul(X.transpose(), X) + lambda_factor*np.identity(n)), np.matmul(X.transpose(), Y))
    return theta
    raise NotImplementedError


# test
    
X = np.array([[0.92611855, 0.95569581],
            [0.91040063, 0.88638017],
            [0.42731193, 0.11492866],
            [0.13942378, 0.70826011],
            [0.34639635, 0.47034816],
            [0.28176313, 0.73971738],
            [0.32380673, 0.12336384],
            [0.36770973, 0.55052563],
            [0.67676161, 0.0612985 ],
            [0.57809415, 0.46286232]])
Y = np.array([0.46005506, 0.96192427, 0.73804515, 0.59918074, 0.14991552, 0.45843984,
 0.05641203, 0.05580739, 0.37788969, 0.24143549])
lambda_factor = 0.12775708203178793

print(closed_form(X, Y, lambda_factor))

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
