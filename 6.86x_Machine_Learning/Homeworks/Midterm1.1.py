import numpy as np
from collections import defaultdict
import sys

def runPerceptron(dataset, theta, theta_0, memo, path, min_len):
    id = -1
    if dataset_id(dataset) not in memo.keys():
        for point in dataset:
            id += 1
            if point[1]*(np.dot(point[0], theta) + theta_0) <= 0:
                theta = theta + point[1]*point[0]
                theta_0 = theta_0 + point[1]

                new_dataset = dataset[:id] + dataset[id+1:]

                path.append(point)
                if new_dataset == []:
                    print("Solution is found!")
                    print("theta:", theta)
                    print("theta_0:", theta_0)
                    print("path:", path)
                    break
                if len(new_dataset) < min_len:
                    min_len = len(new_dataset)
                    #print("number of mistakes achieved:", 50 - min_len)
                else:
                    memo, min_len = runPerceptron(new_dataset, theta, theta_0, memo, path, min_len)
        memo[dataset_id(dataset)] = False
        # if len(memo) % 1000 == 0 and len(memo) != 0:
        #     print("memo:", len(memo))
    return (memo, min_len)

def dataset_id(dataset):
    id_list = []
    for point in dataset:
        id_list.append("_".join([str(point[0][0]), str(point[0][1]), str(point[1])]))
    return ("_").join(id_list)

def path_id(dataset):
    id_list = []
    for point in dataset:
        id_list.append("_".join([str(point[0][0]), str(point[0][1]), str(point[1])]))
    return ("_").join(id_list)
                                  
X = np.array([[0, 0],
              [2, 0], 
              [3, 0],
              [0, 2],
              [2, 2],
              [5, 1],
              [5, 2], 
              [2, 4], 
              [4, 4],
              [5, 5]])

Y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

correct_mistakes = [1, 9, 10, 5, 9, 11, 0, 3, 1, 1]
current_mistakes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

original_dataset = []
dataset = []
for i in range(len(X)):
    #original_dataset.append([X[i], Y[i]])
    # for _ in range(mistakes[i]+1):
    original_dataset.append([X[i], Y[i]])

def perceptron_search(current_mistakes, correct_mistakes, dataset, theta, theta_0, memo, path, max_mistakes):
    #if "_".join([str(i) for i in current_mistakes]) not in memo.keys():
    if True:
        if (all(current_mistakes[i] <= correct_mistakes[i] for i in range(10))):
            new_mistakes = current_mistakes.copy()
            new_path = path.copy()
            id = -1
            for point in dataset:          
                
                id += 1
                if point[1]*(np.dot(point[0], theta) + theta_0) <= 0:
                    new_theta = theta + point[1]*point[0]
                    new_theta_0 = theta_0 + point[1]                     
                    new_mistakes[id] += 1
                    new_path.append(point)
                    if sum(new_mistakes) > max_mistakes:
                        max_mistakes = sum(new_mistakes) 
                    
                    if new_mistakes == correct_mistakes:                    
                        print("Solution is found!")
                        print("theta:", theta)
                        print("theta_0:", theta_0)
                        print("path:", new_path)  
                        print("meistakes:", new_mistakes)
                        sys.exit()
                    elif sum(new_mistakes) < 50:
                        memo, max_mistakes = perceptron_search(new_mistakes, correct_mistakes, dataset, new_theta, new_theta_0, memo, new_path, max_mistakes)
                        memo["_".join([str(i) for i in new_mistakes])] = False
    return (memo, max_mistakes)


memo = defaultdict()

id = -1
for starting_point in original_dataset:
    id += 1
    theta = starting_point[1]*starting_point[0]
    theta_0 = starting_point[1]
    new_path = []
    new_path.append(starting_point)
    mistakes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mistakes[id] += 1
    max_mistakes = 0
    #dataset_key = (tuple(original_dataset[p][0]) for p in range(len(original_dataset)))
    memo, max_mistakes = perceptron_search(mistakes, correct_mistakes, original_dataset, theta, theta_0, memo, new_path, max_mistakes)
    print(max_mistakes)
