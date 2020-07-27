# -*- coding: utf-8 -*-
"""
Created on Sun May 17 21:49:35 2020

@author: dell
"""

import numpy as np
import pandas as pd

class MLP(object):
    def __init__(self, NI, NH, NO):   #     initialise the attributes
        self.no_in = NI               #     NI number of input
        self.no_hidden = NH           #     NH number of Hiddlen units
        self.no_out = NO              #     NO number of output
        self.W1 = np.array        #     An matrix containing the weights in the lower layer
        self.W2 = np.array        #     An matrix containing the weights in the upper layer
        self.dW1 = np.array       #     An matrix containing the weights change to be applied on w1 in the lower layer
        self.dW2 = np.array       #     An matrix containing the weights change to be applied on w2 in the upper layer
        self.Z1 = np.array        #     An array containing the activations in the lower layer
        self.Z2 = np.array        #     An array containing the activations in the upper layer
        self.bias1 = np.array       #     Bias for the lower layer
        self.bias2 = np.array       #     Bias for the upper layer
        self.dBias1 = np.array    #     An matrix containing the bias change to be applied on w1 in the upper layer
        self.dBias2 = np.array    #     An matrix containing the bias change to be applied on w2 in the upper layer
        self.H = np.array  #  array where the values of the hidden neurons are stored – need these saved to compute dW2)
        self.O = np.array  #  array where the outputs are stored   
        
        
    #  Initialize matrix W1 and W2 randomly from Normal distribution having mean 0 and variance 1     
    def randomise(self): 
        self.W1 = np.array((np.random.uniform(low=0, high=1, size=(self.no_in, self.no_hidden))).tolist())
        self.W2 = np.array((np.random.uniform(low=0, high=1, size=(self.no_hidden, self.no_out))).tolist())
        

        # set dW1 and dW2 to all zeroes.
        self.dW1 = np.dot(self.W1, 0)
        self.dW2 = np.dot(self.W2, 0)
        
    # Define a logistic sigmoid function which takes input sigInput and returns 1/(1 + math.exp(-sigInput)).
    def sigmoid(self, sigInput):
        return 1 / (1 + np.exp(-sigInput))
    def derivative_sigmoid(self, sigInput):
        return np.exp(-sigInput) / (1 + np.exp(-sigInput)) ** 2
    
    # Define a logistic tanH function which takes input sigInput and returns 2 / (1 + np.exp(-2*tangInput))-1.
    def tanh(self, tangInput):
        return (2 / (1 + np.exp(tangInput * -2))) - 1
#        return (np.exp(tangInput)-np.exp(-tangInput))/(np.exp(tangInput)+np.exp(-tangInput))
    def derivative_tanH(self, tangInput):
        return 1 - (np.power(self.tanh(tangInput), 2))
        

        
      # Forward pass. Input vector I is processed to produce an output, which is stored in O[].
    def forward(self, I, activation):
    # If we use sigmoid activation function, take the inputs, and put them through the formula to get lower neuron's output
        if activation == 'sigmoid':
            # Array containing the activations in the lower layer
            self.Z1 = np.dot(I, self.W1)
            # Array where the values of the hidden neurons are stored 
            self.H = self.sigmoid(self.Z1)
            
            # Take lower layer's outputs, and put them through the formula to get upper neuron's output
            self.Z2 = np.dot(self.H, self.W2)
            # Array where the outputs are stored
            self.O = self.sigmoid(self.Z2)
    
        elif activation == 'tanh' :
            self.Z1 = np.dot(I, self.W1) 
            # Array where the values of the hidden neurons are stored 
            self.H = self.tanh(self.Z1)
            # If we use tanh activation function,, take lower layer's outputs, and put them through the formula to get upper neuron's output
            self.Z2 = np.dot(self.H, self.W2)
            # Array where the outputs are stored       
            self.O = self.Z2
#             print("tanhforward is" + self.O)
        return self.O
    
    
    #  backward pass 
    #  target is the output that we want, self.O is the output predicted by our network
    def backward(self, I, target, activation):
        output_error = np.subtract(target, self.O) #difference (error) in output 
        if activation == 'sigmoid' : 
            activation_O=self.derivative_sigmoid(self.Z2)
            activation_H=self.derivative_sigmoid(self.Z1)
        elif activation == 'tanh' :
            activation_O=self.derivative_tanH(self.Z2)
            activation_H=self.derivative_tanH(self.Z1)
        dw2_a = np.multiply(output_error, activation_O)
        self.dW2 = np.dot(self.H.T, dw2_a)
        dw1_a=np.multiply(np.dot(dw2_a, self.W2.T), activation_H)
        self.dW1=np.dot(I.T,dw1_a)
        return np.mean(np.abs(output_error))


    # Adjust the weights
    def updateWeights(self, learningRate):
        self.W1 = np.add(self.W1,learningRate * self.dW1)
        self.W2 = np.add(self.W2,learningRate * self.dW2)
        self.dW1 = np.array
        self.dW2 = np.array
