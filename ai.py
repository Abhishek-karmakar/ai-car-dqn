#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:05:57 2018

@author: abhishek

AI for the self driving car
"""

#importing the libraries

import numpy as np
import random
import os #to load the model and 
import torch # implement Neural netork using pytorch
import torch.nn as nn #contain all the tools to implement the neural network. Three signals of the three sensors. 
import torch.nn.functional as F# different functions while using a neural network
import torch.optim as optim # Optimizers to optimize gradinet decent.
import torch.autograd as autograd #put a tensor into a variable containting a tenr and also a gradient.
from torch.autograd import Variable

#creating the architecture of the Neural Network

class Network(nn.Module):
   
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()#to be able to use the functions of the modules
        #variable that is attached to the object 
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) #Full connection between input layer to all the input of hidden layer. 
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state) :
        x = F.relu(self.fc1(state)) # the first full connection of the input neurons
        q_values = self.fc2(x) #get the output Q values of the Neural network
        return q_values # return the Q values.
    
#Imlement Experience Replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] # An empty list. A simple list with 100 elements
        
    def push(self, event):
        self.memory.append(event) #it will append the last event in the memory.
        if len(self.memory) > self.capacity:
            del self.memory[0] # delete the first 
        