# Definition of the Conv architecture: Input-C1-C2-C3-C4-Output
# No MaxPooling - instead strided convolutinos or dilated convolution or both
# Addition of GAP, and FC after GAP

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from matplotlib import pyplot as plt
import numpy as np

## Convolution Model Template
# - Define Model object
# - Define Training module
# - Parameterization to select normalization types

def norm_fn(out_channels, norm_type = "BN", group = 4):
  ''' 
  Returns the normalization to be used based on user inputs.
  User allowed to provide one of 3 inputs: BN | LN | GN for batch layer and group normalization repsectively.
  Specifying groups is only relevant for group_normalization
  '''
  if norm_type == "BN":
    return nn.BatchNorm2d(out_channels)
  elif norm_type == "LN":
    return nn.GroupNorm(1,out_channels) # Put all channels into a single group (equivalent with LayerNorm as per Torch documentation for GroupNorm)
  elif norm_type == "GN":
    return nn.GroupNorm(group, out_channels)
  else:
    print("relevant inputs allowed: BN | LN | GN for batch layer and group normalization repsectively. Proceeding with default batch normalization")
    return nn.BatchNorm2d(out_channels)

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self, dropout_val = 0.1, norm_type = "batch", regularization_type = None, group_GN = 4):
        super(Net, self).__init__()
        self.dropout_val = dropout_val
        self.norm_type = norm_type
        self.group_GN = group_GN
        # Input Block
        self.inp_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.ReLU(),
            norm_fn(32, self.norm_type, self.group_GN),
        ) # Receptive Field = 3 ; 32*32*32

        # convolution block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(1,1), stride=1, padding=0, bias=False),
            nn.ReLU(),
            norm_fn(128, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, groups=128, bias=False), # depthwise separable conv
            nn.ReLU(),
            norm_fn(128, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1,1), stride=1, padding=0, bias=False), # pointwise convolution
            nn.ReLU(),
            norm_fn(32, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
        ) # Receptive Field = 5 ; 32*32*32

        # Transition block 1 - No Max pool ; instead strided convolution ; still early for dilation
        self.transition_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2, padding=1, bias=False), # strided conv
            norm_fn(32, self.norm_type, self.group_GN),
        ) # Receptive Field = 10 ; 32*16*16

        # convolution block 2 with dilaton
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            norm_fn(128, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False), # depthwise separable conv
            nn.ReLU(),
            norm_fn(128, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            norm_fn(32, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
        ) # Receptive Field = 12 ; 32*16*16

        # Transition block 2 ; no max pooling ; strided convolution/ dilation
        self.transition_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False, dilation = 2), # altrous or dilated convolution
            norm_fn(64, self.norm_type, self.group_GN),
        ) # Receptive Field = 24 ; 64*8*8
            
        # convolution block 3 with dilation  
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            norm_fn(128, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False), # depthwise separable conv
            nn.ReLU(),
            norm_fn(128, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            norm_fn(64, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
        ) # Receptive Field = 26 ; 64*8*8

        # Transition block 3
        self.transition_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, bias=False, dilation = 2), # altrous or dilated convolution
            norm_fn(64, self.norm_type, self.group_GN),
        ) # Receptive Field = 52 ; 64*4*4

            
        # consolution block 4       
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            norm_fn(256, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256, bias=False), # depthwise sep conv
            nn.ReLU(),
            norm_fn(256, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            norm_fn(64, self.norm_type, self.group_GN),
            nn.Dropout(self.dropout_val),
        ) # Receptive Field = 54 ; 64*4*4

        # output block # no more relu and dropout for output layers
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # Receptive Field = 108 ; 64*1*1

        self.linear = nn.Linear(256, 10)

    def forward(self, x):
        #C1
        x1 = self.inp_conv(x)
        x2 = self.convblock1(x1)
        x3 = x2 + x1
        #C2
        x4 = self.transition_conv1(x3)
        x5 = self.convblock2(x4)
        x6 = x5 + x4
        #C3
        x7 = self.transition_conv2(x6)
        x8 = self.convblock3(x7)
        x9 = x8 + x7
        #C4
        x10 = self.transition_conv3(x9)
        x11 = self.convblock4(x10)
        #Output
        out = self.gap(x11)        
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return F.log_softmax(out, dim=-1)
