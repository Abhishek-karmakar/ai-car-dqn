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
import os #to load the model and save te model
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
    
    def forward(self, state):
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
        if len(self.memory) > self.capacity: # if the number of elements i larger than the capacity dlete the first element.
            del self.memory[0] # delete the first 
    
    def sample(self, batch_size):
        # if list = ((1,2,3),(4,5,6)), then zip(*list) = ((1,4),(2,3),(5,6))
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #apply the lambda function to all the samples
    
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000) #100K samples in the memory. 
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #lr = learning rate
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #make a fake dimension as a first dimension. 
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): # batch of different states which will become our transition
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.) # to ake sure that we are not deviding by Zero
        
        
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("No Checkpoint found...")
        
        
        