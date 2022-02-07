import numpy as np
import warnings
  
# suppress warnings
warnings.filterwarnings('ignore')

class NeuralNetwork:

    def __init__(self, layer_dims):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.parameters = {}
        L = len(layer_dims) # number of layers in the network

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
            self.parameters['b' + str(l)] = np.random.randn(layer_dims[l], 1) * 0.1
            
            assert self.parameters['W' + str(l)].shape[0] == layer_dims[l], layer_dims[l-1]
            assert self.parameters['W' + str(l)].shape[0] == layer_dims[l], 1
            

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        s = 1./(1+np.exp(-x))
        
        return s
 

    def forward(self, X, parameters):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param parameters: Input vector which is a numpy array.
        :return: Output vector
        """
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        # W3 = parameters["W3"]
        # b3 = parameters["b3"]

        # LINEAR -> SIGMOID -> LINEAR -> SIGMOID -> LINEAR -> SIGMOID
        z1 = np.dot(W1, X) + b1
        a1 = self.activation(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = self.activation(z2)
        # z3 = np.dot(W3, a2) + b3
        # a3 = self.activation(z3)


        return a2

import numpy as np
import warnings
  
# suppress warnings
warnings.filterwarnings('ignore')

class NeuralNetwork:

    def __init__(self, layer_dims):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.parameters = {}
        L = len(layer_dims) # number of layers in the network

        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
            self.parameters['b' + str(l)] = np.random.randn(layer_dims[l], 1) * 0.1
            
            assert self.parameters['W' + str(l)].shape[0] == layer_dims[l], layer_dims[l-1]
            assert self.parameters['W' + str(l)].shape[0] == layer_dims[l], 1
            

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        s = 1./(1+np.exp(-x))
        
        return s
 

    def forward(self, X, parameters):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param parameters: Input vector which is a numpy array.
        :return: Output vector
        """
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        # W3 = parameters["W3"]
        # b3 = parameters["b3"]

        # LINEAR -> SIGMOID -> LINEAR -> SIGMOID -> LINEAR -> SIGMOID
        z1 = np.dot(W1, X) + b1
        a1 = self.activation(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = self.activation(z2)
        # z3 = np.dot(W3, a2) + b3
        # a3 = self.activation(z3)


        return a2
