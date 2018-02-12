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
import torch.nn #contain all the tools to implement the neural network. Three signals of the three sensors. 
import torch.nn.functional as F# different functions while using a neural network
import torch.optim as optim # Optimizers to optimize gradinet decent.
import torch.autograd as autograd #put a tensor into a variable containting a tenr and also a gradient.
from torch.autograd import Variable