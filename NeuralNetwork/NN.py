# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 12:59:41 2018

@author: principal
"""

# neural network class definition 
class neuralNetwork:
    # initialise the neural network
    # to set the number of input, hidden and output nodes
    def __init__(self, inputnodes, hiddennodes, outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate 
        # link weight matrices, wih and who
        # weights inside the arrays are wij, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22
        
        #self.wih=(np.random.rand(self.hnodes,self.inodes)-0.5)# size= hidden_nodesby input_nodes
        #self.who=(np.random.rand(self.onodes,self.hnodes)-0.5)# size= output_nodes by hidden_nodes
        # other way
        self.wih=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function=lambda x: scipy.special.expit(x)
        
        pass
    # train the neural network
    # refine the weights after being given a training set example to learn 
    def train():
        pass
    # query the nerual network
    # give an answer fropm the output nodes after being given an input 
    def query(self,input_list):
        
        pass

# Main program
import numpy as np
import scipy.special
input_nodes=3
hidden_nodes=3
output_nodes=3
learning_rate=0.3
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

