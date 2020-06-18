import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from NN import NN_11, NN_17

from torchsummary import summary

#system_size must be 5
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.net = NN_11(5,3,'cpu')
    
    def forward(self, x):
        x = self.net(x)
        return x

class Critic(nn.Module):
    def __init__(self, num_classes = 3):
        super(Critic, self).__init__()
        self.net = NN_11(5,3,'cpu')
        nc = self.net.linear1.in_features
        self.net.linear1 = nn.Identity()
        self.head = nn.Linear(nc + num_classes, num_classes)
    
    def forward(self, x, a):
        x = self.net(x)
        x = torch.cat((x, a), 1) 
        x = self.head(x)
        return x


'''
def test():
    c = Actor()
    summary(c, (2,5,5,))

test()
'''