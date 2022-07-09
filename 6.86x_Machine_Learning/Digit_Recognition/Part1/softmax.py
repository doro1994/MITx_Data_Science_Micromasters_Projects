import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
#    H = np.empty(theta.shape[0])
#    H = H.reshape(H.shape[0], 1)
#    for x in X:
#        c = max(np.matmul(theta, x.transpose()))/temp_parameter
#        factor = (1/sum(np.exp(np.matmul(theta, x.transpose())/temp_parameter - c))) 
#        h = factor * np.exp(np.matmul(theta, x.transpose())/temp_parameter - c)
#        h = h.reshape(h.shape[0], 1)
#        try:
#            H =np.append(H, h, axis = 1)
#        except:
#            H = h

    #c = max(np.matmul(theta, X.transpose()))/temp_parameter
    n = X.shape[0]
    k = theta.shape[0]
    c = np.matmul(theta, X.transpose())/temp_parameter
    c = np.array(np.max(c, axis = 0)).reshape(1, n)
    c = np.repeat(c, k, axis=0)
    
    factor = np.exp(np.matmul(theta, X.transpose())/temp_parameter - c)
    factor = np.array(1/np.sum(factor, axis = 0)).reshape(1, n)
    factor = np.repeat(factor, k, axis=0)
    
#    factor = (1/sum(np.exp(np.matmul(theta, X.transpose())/temp_parameter - c))) 
#    factor = np.exp(np.matmul(theta, X.transpose())/temp_parameter - c)
    H = factor * np.exp(np.matmul(theta, X.transpose())/temp_parameter - c)
#    h = h.reshape(h.shape[0], 1)    
    return H
    raise NotImplementedError

#test compute_probabilities
n, d, k = 3, 5, 7
X = np.arange(0, n * d).reshape(n, d)
theta = np.arange(0, k * d).reshape(k, d)
temp_parameter = 1
np.matmul(theta, X.transpose())

c = np.matmul(theta, X.transpose())/temp_parameter
c = np.array(np.max(c, axis = 0)).reshape(1, n)
c = np.repeat(c, k, axis=0)

factor = np.exp(np.matmul(theta, X.transpose())/temp_parameter - c)
factor = np.array(1/np.sum(factor, axis = 0)).reshape(1, n)
factor = np.repeat(factor, k, axis=0)

H = compute_probabilities(X, theta, temp_parameter)
a = np.array([[1], [2]])
a = a.reshape(2, 1)
b = np.array([[3], [4]])
b = b.reshape(2, 1)
np.append(b, a, axis = 1)



def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    n = X.shape[0]
    k = theta.shape[0]
    H = compute_probabilities(X, theta, temp_parameter)
    c = 0
    for i in range(n):
        for j in range(k):
            if Y[i] == j:
                c += (-1/n)*np.log(H[j][i])
    c += np.sum(theta**2)*lambda_factor/2
    return c
#    J = sum([np.log(H[j][i]) 
#            for i in range(n)
#            for j in range(k)]
#            if (Y[i] == j))    
    raise NotImplementedError

def run_gradient_descent_iteration_slow(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    n = X.shape[0]
    k = theta.shape[0]
    d = X.shape[1]
    H = compute_probabilities(X, theta, temp_parameter)
    grad_J = np.zeros(k*d).reshape(k, d)
    for m in range(k):
        for i in range(n):
            if Y[i] == m:
                grad_J[m] += (-1/temp_parameter/n)*(X[i]*(1 - H[m][i]))
            else:
                grad_J[m] += (-1/temp_parameter/n)*(X[i]*(-H[m][i])) 
        grad_J[m] += lambda_factor*theta[m]
    return theta - alpha*grad_J  
    raise NotImplementedError

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    itemp=1./temp_parameter
    num_examples = X.shape[0]
    num_labels = theta.shape[0]
    probabilities = compute_probabilities(X, theta, temp_parameter)
    # M[i][j] = 1 if y^(j) = i and 0 otherwise.
    M = sparse.coo_matrix(([1]*num_examples, (Y,range(num_examples))), shape=(num_labels,num_examples)).toarray()
    non_regularized_gradient = np.dot(M-probabilities, X)
    non_regularized_gradient *= -itemp/num_examples
    return theta - alpha * (non_regularized_gradient + lambda_factor * theta)

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    train_y_mod3 = np.array([(y % 3) for y in train_y])
    test_y_mod3 = np.array([(y % 3) for y in test_y])
    return (train_y_mod3, test_y_mod3)
    raise NotImplementedError

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    pred_Y = get_classification(X, theta, temp_parameter)
    pred_Y = [y % 3 for y in pred_Y]
    return 1 - np.mean(pred_Y == Y)
    raise NotImplementedError
    


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
