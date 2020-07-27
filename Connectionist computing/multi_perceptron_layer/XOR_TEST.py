# -*- coding: utf-8 -*-
"""
Created on Sun May 17 21:50:23 2020

@author: dell
"""
import numpy as np
import pandas as pd
from MLP import MLP

log = open("xortest.txt", "w")
print("XOR TEST\n", file = log)


def XOR(max_epochs, learning_rate):
    np.random.seed(1)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    NI = 2
    NH = 4
    NO = 1
    NN = MLP(NI, NH, NO)

    NN.randomise()
    print('\nMax Epoch:\n' + str(max_epochs), file=log)
    print('\nLearning Rate:\n' + str(learning_rate), file=log)
    print('\nBefore Training:\n', file=log)
    for i in range(len(inputs)):
        NN.forward(inputs[i],'sigmoid')
        print('Target:\t {}  Output:\t {}'.format(str(outputs[i]), str(NN.O)), file=log)
    print('\nTraining:\n', file=log)
    
    for i in range(0, max_epochs):
        NN.forward(inputs,'sigmoid')
        error = NN.backward(inputs, outputs,'sigmoid')
        NN.updateWeights(learning_rate)

        if (i + 1) % (max_epochs / 20) == 0:
            print(' Error at Epoch:\t' + str(i + 1) + '\t\t  is \t\t' + str(error), file=log)

    print('\n After Training :\n', file=log)
    
    accuracy=float(0)
    for i in range(len(inputs)):
        NN.forward(inputs[i],'sigmoid')
        print('Target:\t {}  Output:\t {}'.format(str(outputs[i]), str(NN.O)), file=log)
        if(outputs[i][0]==0):
            accuracy+=1-NN.O[0]
        elif(outputs[i][0]==1):
            accuracy+=NN.O[0]
    print('\nAccuracy:{}'.format(accuracy/4),file=log)
iteration=[10000,1000]
learn_rate=[1.0,0.8,0.6,0.4,0.2,0.02]

for i in range(len(iteration)):
    for j in range(len(learn_rate)):
        print('----------------------------------------------------------------------\n', file=log)
        XOR(iteration[i],learn_rate[j])
        print('\n-------------------------------------------------------------------\n', file=log)
