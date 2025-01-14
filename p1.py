import numpy as np


np.random.seed(0)

X = [[1,2,3,2.5], # Best practice to use X as inputs variable
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]




class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #using 0.10 to ensure weights are small numbers
        self.biases = np.zeros((1,n_neurons)) #Use the shape here
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU: #Activation Function
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(4,5) 
layer2 = Layer_Dense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)



print(layer2.output)
