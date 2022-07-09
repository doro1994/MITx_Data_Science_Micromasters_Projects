import numpy as np
import math
import tqdm

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return max(0, x)

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    if x > 0:
        return 1
#    elif x < 0:
#        return 0
    else:
        #print("DERIVATIVE NOT EXISTS")
        return 0

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1.0 1.0; 1.0 1.0; 1.0 1.0')
        self.hidden_to_output_weights = np.matrix('1.0 1.0 1.0')
        self.biases = np.matrix('0.0; 0.0; 0.0')
        self.learning_rate = .001
        self.epochs_to_train = 10
        #self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.training_points = [((-2, 1), -1), 
                                ((2, -2), 0), 
                                ((7, -7), 0), 
                                ((3, 2), 5), 
                                ((4, -5), -1), 
                                ((-6, 7), 1), 
                                ((-8, -5), -13), 
                                ((2, 7), 9), 
                                ((-5, -1), -6), 
                                ((-6, -2), -8)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10), (2,1), (-1, -2), (0, 3)]
        


    def train(self, x1, x2, y):
        
        x1 = float(x1)
        x2 = float(x2)
        y = float(y)
        self.biases = np.float_(self.biases)
        self.hidden_to_output_weights = np.float_(self.hidden_to_output_weights)
        self.input_to_hidden_weights = np.float_(self.input_to_hidden_weights)

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.matmul(self.input_to_hidden_weights, input_values) + self.biases #  (3 by 1 matrix)
        hidden_layer_activation = np.vectorize(rectified_linear_unit)(hidden_layer_weighted_input) # (3 by 1 matrix)

        output = np.matmul(self.hidden_to_output_weights, hidden_layer_activation) 
        activated_output = np.vectorize(output_layer_activation)(output) 
        
        output_layer_error = - (y - activated_output)*output_layer_activation_derivative(output)

        ### Backpropagation ###

        # Compute gradients
        f_ReLU_prime_hidden = np.float_(np.vectorize(rectified_linear_unit_derivative)(hidden_layer_weighted_input))
        hidden_layer_error = np.multiply(output_layer_error, self.hidden_to_output_weights.transpose()) # (3 by 1 matrix)
        hidden_layer_error = np.multiply(f_ReLU_prime_hidden, hidden_layer_error)
       
        bias_gradients = hidden_layer_error # TODO
        hidden_to_output_weight_gradients  = np.multiply(output_layer_error, hidden_layer_activation) # TODO
        input_to_hidden_weight_gradients = np.matmul(hidden_layer_error, input_values.transpose()) # TODO

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases - self.learning_rate * bias_gradients # TODO
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate * input_to_hidden_weight_gradients # TODO
        self.hidden_to_output_weights = self.hidden_to_output_weights  - self.learning_rate * hidden_to_output_weight_gradients.transpose() # TODO
        print(f"(Input --> Hidden Layer) Weights: ({x1},{x2}) --> {self.input_to_hidden_weights}" )
        print(f"Biases: ({x1},{x2}) --> {self.biases}" )
 # My output       
#Epoch  0
#(Input --> Hidden Layer) Weights:  [[1.17202554 0.8718929 ]
# [1.17202554 0.8718929 ]
# [1.17202554 0.8718929 ]]
#(Hidden --> Output Layer) Weights:  [[0.56889639 0.56889639 0.56889639]]
#Biases:  [[-0.09733494]
# [-0.09733494]
# [-0.09733494]]
#Epoch  1
#(Input --> Hidden Layer) Weights:  [[1.28605651 0.95254752]
# [1.28605651 0.95254752]
# [1.28605651 0.95254752]]
#(Hidden --> Output Layer) Weights:  [[0.42280957 0.42280957 0.42280957]]
#Biases:  [[-0.13795895]
# [-0.13795895]
# [-0.13795895]]        

# Correct output    
#Epoch  0
#(Input --> Hidden Layer) Weights:  [[0.91092697 0.7643996 ]
# [0.91092697 0.7643996 ]
# [0.91092697 0.7643996 ]]
#(Hidden --> Output Layer) Weights:  [[0.66407787 0.66407787 0.66407787]]
#Biases:  [[-0.04794291]
# [-0.04794291]
# [-0.04794291]]
#Epoch  1
#(Input --> Hidden Layer) Weights:  [[0.86094333 0.74126222]
# [0.86094333 0.74126222]
# [0.86094333 0.74126222]]
#(Hidden --> Output Layer) Weights:  [[0.56676748 0.56676748 0.56676748]]
#Biases:  [[-0.06026656]
# [-0.06026656]
# [-0.06026656]]    
        
 
b = np.matrix('1.0 1.0 1.0')
c = np.matrix('2.0 2.0 2.0')
d = np.matrix('3.0 3.0 3.0')
a = np.multiply(b, c, d)
assert a == np.matrix('2.0 2.0 2.0')
assert d == np.matrix('2.0 2.0 2.0')
        
#        
#x1 = 7
#x2 = -7
#y = 0
#
#input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1')
#hidden_to_output_weights = np.matrix('1 1 1')
#biases = np.matrix('0; 0; 0')
#learning_rate = .001
#
#x1 = float(x1)
#x2 = float(x2)
#y = float(y)
#
#biases = np.float_(biases)
#hidden_to_output_weights = np.float_(hidden_to_output_weights)
#input_to_hidden_weights = np.float_(input_to_hidden_weights)
#
#### Forward propagation ###
#input_values = np.matrix([[x1],[x2]]) # 2 by 1
#
## Calculate the input and activation of the hidden layer
#hidden_layer_weighted_input = np.matmul(input_to_hidden_weights, input_values) + biases #  (3 by 1 matrix)
#hidden_layer_activation = np.vectorize(rectified_linear_unit_derivative)(hidden_layer_weighted_input) # (3 by 1 matrix)
#
#output = np.matmul(hidden_to_output_weights, hidden_layer_activation) 
#activated_output = np.vectorize(output_layer_activation)(output) 
#
#### Backpropagation ###
#
## Compute gradients
#output_layer_error = - (y - activated_output)*output_layer_activation_derivative(output)
#f_ReLU_prime_hidden = np.float_(np.vectorize(rectified_linear_unit_derivative)(hidden_layer_weighted_input))
#hidden_layer_error = np.multiply(output_layer_error, hidden_to_output_weights.transpose()) # (3 by 1 matrix)
#hidden_layer_error = np.multiply(f_ReLU_prime_hidden, hidden_layer_error) # (3 by 1 matrix)
#bias_gradients = hidden_layer_error # TODO
#hidden_to_output_weight_gradients  = np.multiply(output_layer_error, hidden_layer_activation) # TODO
#input_to_hidden_weight_gradients = np.matmul(hidden_layer_error, input_values.transpose()) # TODO
#
#biases = biases - learning_rate * bias_gradients # TODO
#input_to_hidden_weights = input_to_hidden_weights- learning_rate * input_to_hidden_weight_gradients # TODO
#hidden_to_output_weights = hidden_to_output_weights - learning_rate * hidden_to_output_weight_gradients.transpose()# TODO


    def predict(self, x1, x2):

        input_values = np.matrix([[x1],[x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = np.matmul(self.input_to_hidden_weights, input_values) + self.biases # TODO
        hidden_layer_activation = np.vectorize(rectified_linear_unit)(hidden_layer_weighted_input) # TODO
        output = np.matmul(self.hidden_to_output_weights, hidden_layer_activation)  # TODO
        activated_output = np.vectorize(output_layer_activation)(output)  # TODO

        return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):
        with tqdm.trange(self.epochs_to_train) as t:
            for epoch in t:
                for x,y in self.training_points:
                    self.train(x[0], x[1], y)
                #print("Epoch", epoch)
                #print("(Input --> Hidden Layer) Weights:", self.input_to_hidden_weights )
                #break
                

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            #if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
            if abs(self.predict(point[0], point[1]) - (point[0] + point[1])) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()
