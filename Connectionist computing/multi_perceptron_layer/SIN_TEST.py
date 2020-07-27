# -*- coding: utf-8 -*-
"""
Created on Sun May 17 21:56:52 2020

@author: dell
"""
import numpy as np
import pandas as pd
from MLP import MLP

log = open("sintest.txt", "w")
print("SINE TEST\n", file = log)

def SIN(max_epochs, learning_rate, no_hidden):
    np.random.seed(213)
    inputs = []
    outputs = []
    for i in range(0, 200):
        four_inputs_vector = list(np.random.uniform(-1.0, 1.0, 4))
        four_inputs_vector = [float(four_inputs_vector[0]),float(four_inputs_vector[1]), 
                              float(four_inputs_vector[2]),float(four_inputs_vector[3])]
        inputs.append(four_inputs_vector)
    inputs=np.array(inputs)

    for i in range(200):
        outputs.append(np.sin([inputs[i][0] - inputs[i][1] + inputs[i][2] - inputs[i][3]]))

    no_in = 4
    no_out = 1
    NN = MLP(no_in, no_hidden, no_out)
    NN.randomise()
    print('\nMax Epoch:\n' + str(max_epochs), file=log)
    print('\nLearning Rate:\n' + str(learning_rate), file=log)
    print('\nBefore Training:\n', file=log)

    for i in range(150):
        NN.forward(inputs[i],'tanh')
        print('Target:\t{}\t Output:\t {}'.format(str(outputs[i]),str(NN.O)), file=log)
    print('Training:\n', file=log)
    
    
#    training process
    for i in range(0, max_epochs):
        error = 0
        NN.forward(inputs[:150],'tanh')
        error = NN.backward(inputs[:150], outputs[:150],'tanh')
        NN.updateWeights(learning_rate)
       #prints error every 5% of epochs
        if (i + 1) % (max_epochs / 20) == 0:
            print(' Error at Epoch:\t' + str(i + 1) + '\t  is \t' + str(error), file=log)
    
    difference=float(0)
    print('\n Testing :\n', file=log)
    for i in range(150, len(inputs)):
        NN.forward(inputs[i], 'tanh')
        print('Target:\t{}\t Output:\t {}'.format(str(outputs[i]), str(NN.O)), file=log)
        difference+=np.abs(outputs[i][0]-NN.O[0])

    accuracy=1-(difference/50)
    accuracylist.append(accuracy)
    print('\nAccuracy:{}'.format(accuracy),file=log)
    print('\ntestError:{}'.format(difference/50),file=log)
    

iteration=[100000]
learn_rate=[0.02,0.001,0.0006,0.0001]
accuracylist=[]
for i in range(len(iteration)):
    for j in range(len(learn_rate)):
         print('----------------------------------------------------------------------\n', file=log)
         SIN(iteration[i], learn_rate[j], no_hidden=10)
         print('\n-------------------------------------------------------------------\n', file=log)
print('Accuracylist:{}'.format(accuracylist),file=log)
