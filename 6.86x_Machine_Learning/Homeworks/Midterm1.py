# Problem 1

import numpy as np
import random
from itertools import permutations
import matplotlib.pyplot as plt
import tqdm 

import matplotlib.lines as lines




X = np.array([[0, 0],
              [2, 0], 
              [3, 0],
              [0, 2],
              #[4, 3.5],
              [2, 2],
              [5, 1],
              [5, 2], 
              [2, 4], 
              [4, 4],
              [5, 5]])

Y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])


X_cycle = np.repeat(X, 100)
Y_cycle = np.repeat(Y, 100)

correct_mistakes = [1, 9, 10, 5, 9, 11, 0, 3, 1, 1]
mistakes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def perceptron_single_update(x: np.array, theta: np.array, theta_0: float, y: int, id):
    if np.dot(x, theta)*y < 0:
        theta = theta + y*x
        theta_0 = theta_0 + y
        mistakes[id] += 1
        
T = 1000       

#theta = np.random.rand(2)
#theta_0 = np.random.randint(-20, 20+1)
#theta_init = theta
#theta_0_init = theta_0

#indices = [1, 3, 4, 5, 7, 9, 6, 2, 8, 0]
# with tqdm.trange(10**10) as tt:
#     for i in tt:
#         mistakes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
#         theta = np.array([0.0, 0.0])
#         theta_0 = 0
#         indices = list(range(0, 10)) 
        
#         for t in range(T):
#             random.shuffle(indices)
#             for id in indices:
#                 x = X[id]
#                 y = Y[id]            
#                 if (np.dot(x, theta) + theta_0)*y <= 0:
#                     theta = theta + y*x
#                     theta_0 = theta_0 + y
#                     mistakes[id] += 1
    
# #            if len(mistakes) > 52:
# #                print("more thank 52 mistakes!")
                    
#             if (mistakes[0] > correct_mistakes[0]
#                 or mistakes[1] > correct_mistakes[1]
#                 or mistakes[2] > correct_mistakes[2]
#                 or mistakes[3] > correct_mistakes[3]
#                 or mistakes[4] > correct_mistakes[4]
#                 or mistakes[5] > correct_mistakes[5]
#                 or mistakes[6] > correct_mistakes[6]
#                 or mistakes[7] > correct_mistakes[7]
#                 or mistakes[8] > correct_mistakes[8]
#                 or mistakes[9] > correct_mistakes[9]):
#                 break

        
#         if len(mistakes) == 50:
#             print("50 mistakes! Theta:", theta, ", theta_0:", theta_0, mistakes)    
#         if mistakes == correct_mistakes:
#             print("Found! Theta:", theta, ", theta_0:", theta_0, mistakes)
#         if i < 10:
#             print("Example! Theta:", theta, ", theta_0:", theta_0, mistakes)    
#        else:
            #print("Not Found! Theta:", theta, ", theta_0:", theta_0, mistakes)
#            pass


#Loss = 0  
#theta = np.array([1, 1])
#theta_0 = -10 
#for id in range(10):
#    Loss += max(0, 1 - (np.dot(theta, X[id]) + theta_0) * Y[id])    
#    print("Point", X[id], ":", max(0, 1 - (np.dot(theta, X[id])  + theta_0) * Y[id]))
#print("Loss:", Loss)  
#
#dataset = np.array([[0, 0]])
#dataset = np.append(dataset, np.tile(X[1], (9, 1)), axis = 0)
#dataset = np.append(dataset, np.tile(X[2], (10, 1)), axis = 0)
#dataset = np.append(dataset, np.tile(X[3], (5, 1)), axis = 0)
#dataset = np.append(dataset, np.tile(X[4], (9, 1)), axis = 0)
#dataset = np.append(dataset, np.tile(X[5], (11, 1)), axis = 0)
#dataset = np.append(dataset, np.tile(X[7], (3, 1)), axis = 0)
#dataset = np.append(dataset, np.tile(X[8], (1, 1)), axis = 0)
#dataset = np.append(dataset, np.tile(X[9], (1, 1)), axis = 0)
#print(len(dataset))
#
#index_matrix = np.random.permutation(list(range(50)))
#
#print("Process has been started")
#for _ in range(1000000):
#    mistakes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#    theta = np.array([0.0, 0.0])
#    theta_0 = 0.0 
#    start_id = id
#    indices = np.random.permutation(list(range(50)))
#    for id in indices:
#        x = dataset[id]
#        y = dataset[id]            
#        if (np.dot(x, theta) + theta_0)*y <= 0:
#            theta = theta + y*x
#            theta_0 = theta_0 + y
#            mistakes[id] += 1
#
#
#    if mistakes == correct_mistakes:
#        print("Found! Theta:", theta, ", theta_0:", theta_0, mistakes, dataset[id])
#    else:
#        #print("Not Found! Theta:", theta, ", theta_0:", theta_0, mistakes) 
#        pass
#print("Process has been finished")
#    
    
    
def perceptron_single_step_update(
    feature_vector,
    label,
    current_theta,
    current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.
    
    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    
    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    if label * (np.matmul(feature_vector, current_theta) + current_theta_0) <= 0:
        current_theta = current_theta + label*feature_vector
        current_theta_0 = current_theta_0 + label
    return (current_theta, current_theta_0)
    raise NotImplementedError

def line_create(theta, theta_0, fig): 
    theta_1 = theta[0]
    theta_2 = theta[1]
    if theta_1 == 0:
        theta_1 = 0.00001
    if theta_2 ==0:
        theta_2 = 0.00001
    x11 = 0
    x21 = (-x11*theta_1 - theta_0) / theta_2
    x12 = 1
    x22 = (-x12*theta_1 - theta_0) / theta_2    
    line = plt.axline((x11, x21), (x12, x22), linewidth=2, color='black') 
    return line
    
def line_update(theta, theta_0, fig, line): 
    line.remove()
    theta_1 = theta[0]
    theta_2 = theta[1]
    if theta_1 == 0:
        theta_1 = 0.00001
    if theta_2 ==0:
        theta_2 = 0.00001
    x11 = 0
    x21 = (-x11*theta_1 - theta_0) / theta_2
    x12 = 1
    x22 = (-x12*theta_1 - theta_0) / theta_2    
    line = plt.axline((x11, x21), (x12, x22), linewidth=2, color='black')    
    return line

# theta = np.array([2, 1])
# theta_0 = 5

#Generate the Scatterplot
# plot_scatter(X,Y)    

#def plot_scatter(X,y):

# Plot the points    
colors = ["blue","red","black","yellow","green","purple","orange"]
markers = ('o', 'o', 'o', '^', 'v')
fig = plt.figure()

for i, yi in enumerate(np.unique(Y)):
    Xi = X[Y==yi]
    plt.scatter(Xi[:,0], Xi[:,1],
                    color=colors[i], marker=markers[i], label=yi)

plt.xlabel('X label')
plt.ylabel('Y label')
plt.legend(loc='upper left')

plt.xlim(-1, 6)
plt.ylim(-1, 6)

plt.ion()
    
theta = np.array([0, 0])
theta_0 = 0
line = line_create(theta, theta_0, fig)

# feed the first point
theta, theta_0 = perceptron_single_step_update(np.array([2, 2]), -1, theta, theta_0)
line = line_update(theta, theta_0, fig, line)

theta, theta_0 = perceptron_single_step_update(np.array([3, 0]), -1, theta, theta_0)
line = line_update(theta, theta_0, fig, line)

theta, theta_0 = perceptron_single_step_update(np.array([3, 4]), 1, theta, theta_0)
line = line_update(theta, theta_0, fig, line)



