import numpy
from utilities import *

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs):
        # Weight inputs, add bias, then use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)