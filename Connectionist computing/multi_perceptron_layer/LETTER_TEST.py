# -*- coding: utf-8 -*-
"""
Created on Sun May 17 21:57:34 2020

@author: dell
"""
import numpy as np
import pandas as pd
from MLP import MLP

log = open("letter.txt", "w")
print("letter recognition test\n", file = log)

def letter(max_epochs, learning_rate):
    np.random.seed(1)
 
    inputs = []
    outputs = []
    doutput = []
    columns=["letter","x-box","y-box","width","height","onpix","x-bar","y-bar","x2bar","y2bar","xybar","x2ybr","xy2br","x-ege","xegvy","y-ege","yegvx"]
    
    df=pd.read_csv("letter-recognition.data", names=columns)
    doutput=df["letter"]
    
    
    for i in range(len(doutput)):
        outputs.append(ord(str(doutput[i]))-ord('A'))
    
    inputs=df.drop(["letter"], axis=1)
    inputs=np.array(inputs)
    inputs=inputs/15  #normalization
    
    #train set
    inputs_train=inputs[:16000]
    categorical_y = np.zeros((16000, 26))
    for i, l in enumerate(outputs[:16000]):
        categorical_y[i][l] = 1
    outputs_train=categorical_y
    
    #test set
    inputs_test=inputs[16000:]
#    categorical_y = np.zeros((4000, 26))
#    for i, l in enumerate(outputs[16000:]):
#        categorical_y[i][l] = 1
#    outputs_test=categorical_y
    
    #training process
    no_in= 16
    no_hidden = 10
    no_out = 26
    
    NN = MLP(no_in, no_hidden, no_out)
    NN.randomise()
    print('\nMax Epoch:\n' + str(max_epochs), file=log)
    print('\nLearning Rate:\n' + str(learning_rate), file=log)
    print('\nTraining Process:\n', file=log)
    
    for i in range(0, max_epochs):
        NN.forward(inputs_train,'tanh')
        error = NN.backward(inputs_train, outputs_train,'tanh')
        NN.updateWeights(learning_rate)
    
        if (i + 1) % (max_epochs / 20) == 0:
            print(' Error at Epoch:\t' + str(i + 1) + '\t  is \t' + str(error),file=log)
    
    
    #testing process
    def to_character0(outputvector):
        listov=list(outputvector)
        a=listov.index(max(listov))
        return chr(a+ord('A'))
    
    prediction=[]
    for i in range(4000):
        NN.forward(inputs_test[i],'tanh')
    #    print('Target:\t{}\t Output:\t{}'.format(str(outputs_test[i]),str(NN.O)))
    #    print('Target:\t{}\t Output:\t{}'.format(str(doutput[16000+i]),str(to_character0(NN.O))))
        prediction.append(to_character0(NN.O))
    
    
    
    def to_character(n):
        return chr(int(n) + ord('A'))
    
    correct = {to_character(i): 0 for i in range(26)}
    letter_num = {to_character(i): 0 for i in range(26)}
    
    print('==' * 30,file=log)
    for i, _ in enumerate(doutput[16000:]):
        letter_num[doutput[16000+i]] += 1
        # Print some predictions
        if i % 300 == 0:
            print('Expected: {} | Output: {}'.format(doutput[16000+i], prediction[i]),file=log)
        if doutput[16000+i] == prediction[i]:
            correct[prediction[i]] += 1
    
    print('==' * 30,file=log)
    # Calculate the accuracy
    accuracy = sum(correct.values()) / len(prediction)
    print('Test sample size: {} | Correctly predicted sample size: {}'.format(len(prediction),sum(correct.values())),file=log) 
    print('Accuracy: %.3f' % accuracy,file=log)
    
    # Performance on each class
    print('==' * 30,file=log)
    for k,v in letter_num.items():
        print('{} => Sample Number: {} | Correct Number: {} | Accuracy: {}'.format(k, v, correct[k], correct[k]/v),file=log)


iteration=[100000]
learn_rate=[0.000005]
for i in range(len(iteration)):
    for j in range(len(learn_rate)):
         print('----------------------------------------------------------------------\n', file=log)
         letter(iteration[i], learn_rate[j])
         print('\n-------------------------------------------------------------------\n', file=log)
