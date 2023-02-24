# Custom Resnet
import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual block
class ResidualBlock(nn.Module):
  '''
  Residualblock => (Conv-BN-ReLU-Conv-BN-ReLU))(X)
  '''
  def __init__(self, num_channel):
      super(ResidualBlock, self).__init__()
      self.conv1 = nn.Sequential(
                            nn.Conv2d(num_channel, num_channel, 3, padding=1),
                            nn.BatchNorm2d(num_channel),
                            nn.ReLU())
      self.conv2 = nn.Sequential(
                            nn.Conv2d(num_channel, num_channel, 3, padding=1),
                            nn.BatchNorm2d(num_channel),
                            nn.ReLU())
      self.relu = nn.ReLU()

  def forward(self, x):
      residual = x
      x = self.conv1(x)
      x = self.conv2(x)
      out = x + residual
      return out

# layerblock
class LayerBlock(nn.Module):
  '''
  Layer block:
  conv_pool = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU
  Residualblock = ResBlock((Conv-BN-ReLU-Conv-BN-ReLU))(X)
  Add(X, R)
  '''
  def __init__(self, channel_in, channel_out):
      super(LayerBlock, self).__init__()

      self.conv_pool = nn.Sequential(
          nn.Conv2d(channel_in, channel_out, 3, 1, padding=1),
          nn.MaxPool2d(2, 2),
          nn.BatchNorm2d(channel_out),
          nn.ReLU(),
      )

      self.residualblock = ResidualBlock(channel_out)

  def forward(self, x):
      X = self.conv_pool(x)
      R = self.residualblock(X)
      out = X + R
      return out

# Custom ResNet main class
class Custom_ResNet(nn.Module):
  '''
  PrepLayer => Layer1 (LayerBlock) => Layer2 (conv_pool) => Layer3 (LayerBlock) => Layer4 (maxpool) => Layer5 (FC) => softmax
  '''
  def __init__(self, output_classes=10):
      super(Custom_ResNet, self).__init__()

      self.preplayer = nn.Sequential(
          nn.Conv2d(3, 64, 3, 1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
      ) # RF = 3 (j=1)

      self.layer1 = LayerBlock(64, 128) # RF = 5-> 7 (j=2)
      
      self.layer2 = nn.Sequential(
          nn.Conv2d(128, 256, 3, 1, padding=1),
          nn.MaxPool2d(2, 2),
          nn.BatchNorm2d(256),
          nn.ReLU(),
      )  # RF = 11 -> 15 (j=4)
      
      self.layer3 = LayerBlock(256, 512) # RF = 23 -> 31 (j=8)
            
      self.linear = nn.Linear(512, output_classes) # FC layer

  def forward(self, x):
      out = self.preplayer(x)
      out = self.layer1(out)
      out = self.layer2(out)
      out = self.layer3(out)
      out = F.avg_pool2d(out,4) # RF = 31 + (4-1)*8 = 55
      out = out.view(out.size(0), -1)
      out = self.linear(out)
      return F.log_softmax(out, dim=-1)
