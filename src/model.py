import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from NN import NN_11, NN_17


##まだResNetにしか対応してない
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, Network):
        super(Actor, self).__init__()
        self.net = Network()
    
    def forward(self, x):
        x = self.net(x)
        return x

class Critic(nn.Module):
    def __init__(self, Network, num_classes = 4):
        super(Critic, self).__init__()
        self.net = Network()
        nc = self.net.linear.in_features
        self.net.linear = nn.Identity()
        self.head = nn.Linear(nc + num_classes, num_classes)
    
    def forward(self, x, a):
        x = self.net(x)
        x = torch.cat((x, a), 1)
        x = self.head(x)
        return x