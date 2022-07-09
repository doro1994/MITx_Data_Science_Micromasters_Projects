# Project 0
import numpy as np
def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x <= y:
        return x*y
    else:
        return x/y
    raise NotImplementedError
scalar_function(3,2)

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    vect_func = np.vectorize(scalar_function)
    return vect_func(x, y)

    raise NotImplementedError
    
x = [1, 2]
y = [2]
vector_function(x, y)

def get_sum_metrics(predictions, metrics=None):
    #import pdb; pdb.set_trace()
    #metrics=[]
    if metrics is None:
        metrics = []

    for i in range(3):
        def metric(x, j=i):
            return x + j        
        metrics.append(metric)
    sum_metrics = 0
    for k in range(len(metrics)):
        sum_metrics += metrics[k](predictions)
        
    return sum_metrics

print(get_sum_metrics(2))

print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9
print(get_sum_metrics(3, [lambda x: x]))  # Should be (3) + (3 + 0) + (3 + 1) + (3 + 2) = 15
print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9