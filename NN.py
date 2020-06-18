import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.util import conv_to_fully_connected, pad_circular

from torchsummary import summary


'''
input:
    system_size : 格子のサイズ
    number_of_actions : 行動の種類(出力サイズの指定に用いる)
    device : cpu or cuda
'''

# neural network CNN with one fully connected layer
class NN_11(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_11, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 120, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(120, 111, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(111, 104, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(104, 103, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(103, 90, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(90, 80 , kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(80, 73 , kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(73, 71 , kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(71, 64, kernel_size=3, stride=1)
        output_from_conv = conv_to_fully_connected(system_size, 3, 0, 1)
        self.linear1 = nn.Linear(64*int(output_from_conv)**2, 3)
        self.device = device

    def forward(self, x):
        x = pad_circular(x, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = self.linear1(x)
        return x


class NN_17(nn.Module):

    def __init__(self, system_size, number_of_actions, device):
        super(NN_17, self).__init__()
        self.conv1 = nn.Conv2d(2, 256, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 251, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(251, 250, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(250, 240, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(240, 240, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(240, 235, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(235, 233, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(233, 233, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(233, 229, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(229, 225, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(225, 223, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(223, 220 , kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(220, 220 , kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(220, 220 , kernel_size=3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(220, 215 , kernel_size=3, stride=1, padding=1)
        self.conv17 = nn.Conv2d(215, 214 , kernel_size=3, stride=1, padding=1)
        self.conv18 = nn.Conv2d(214, 205 , kernel_size=3, stride=1, padding=1)
        self.conv19 = nn.Conv2d(205, 204 , kernel_size=3, stride=1, padding=1)
        self.conv20 = nn.Conv2d(204, 200 , kernel_size=3, stride=1)
        output_from_conv = conv_to_fully_connected(system_size, 3, 0, 1)
        self.linear1 = nn.Linear(200*int(output_from_conv)**2, number_of_actions)
        self.device = device

    def forward(self, x):
        x = pad_circular(x, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x = F.relu(self.conv19(x))
        x = F.relu(self.conv20(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = self.linear1(x)
        return x

#def test():
#    model = NN_11(5,3,'cpu')
#    summary(model,(2,5,5))

#test()


'''
NN_11
system_size = 5
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 128, 5, 5]           2,432
            Conv2d-2            [-1, 128, 5, 5]         147,584
            Conv2d-3            [-1, 120, 5, 5]         138,360
            Conv2d-4            [-1, 111, 5, 5]         119,991
            Conv2d-5            [-1, 104, 5, 5]         104,000
            Conv2d-6            [-1, 103, 5, 5]          96,511
            Conv2d-7             [-1, 90, 5, 5]          83,520
            Conv2d-8             [-1, 80, 5, 5]          64,880
            Conv2d-9             [-1, 73, 5, 5]          52,633
           Conv2d-10             [-1, 71, 5, 5]          46,718
           Conv2d-11             [-1, 64, 3, 3]          40,960
           Linear-12                    [-1, 3]           1,731
================================================================
Total params: 899,320
Trainable params: 899,320
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.20
Params size (MB): 3.43
Estimated Total Size (MB): 3.63
----------------------------------------------------------------
'''


'''
NN_17
system_size = 5
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 256, 5, 5]           4,864
            Conv2d-2            [-1, 256, 5, 5]         590,080
            Conv2d-3            [-1, 251, 5, 5]         578,555
            Conv2d-4            [-1, 250, 5, 5]         565,000
            Conv2d-5            [-1, 240, 5, 5]         540,240
            Conv2d-6            [-1, 240, 5, 5]         518,640
            Conv2d-7            [-1, 235, 5, 5]         507,835
            Conv2d-8            [-1, 233, 5, 5]         493,028
            Conv2d-9            [-1, 233, 5, 5]         488,834
           Conv2d-10            [-1, 229, 5, 5]         480,442
           Conv2d-11            [-1, 225, 5, 5]         463,950
           Conv2d-12            [-1, 223, 5, 5]         451,798
           Conv2d-13            [-1, 220, 5, 5]         441,760
           Conv2d-14            [-1, 220, 5, 5]         435,820
           Conv2d-15            [-1, 220, 5, 5]         435,820
           Conv2d-16            [-1, 215, 5, 5]         425,915
           Conv2d-17            [-1, 214, 5, 5]         414,304
           Conv2d-18            [-1, 205, 5, 5]         395,035
           Conv2d-19            [-1, 204, 5, 5]         376,584
           Conv2d-20            [-1, 200, 3, 3]         367,400
           Linear-21                    [-1, 3]           5,403
================================================================
Total params: 8,981,307
Trainable params: 8,981,307
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.85
Params size (MB): 34.26
Estimated Total Size (MB): 35.11
----------------------------------------------------------------
'''