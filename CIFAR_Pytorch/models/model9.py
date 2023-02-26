# self attention model using ultimus blocks 
# https://canvas.instructure.com/courses/5720700/assignments/35638794?module_item_id=80344418
import torch
import torch.nn as nn
import torch.nn.functional as F

class UltimusBlock(nn.Module):
'''
Created a block called ULTIMUS that:
Created 3 FC layers called K, Q and V such that:
X*K = 48*48x8 > 8
X*Q = 48*48x8 > 8 
X*V = 48*48x8 > 8 
then created AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
then Z = V*AM = 8*8 > 8
then another FC layer called Out that:
Z*Out = 8*8x48 > 48
'''
  def __init__(self):
    super().__init__()
    self.keylayer = nn.Linear(48, 8)
    self.querylayer = nn.Linear(48,8)
    self.valuelayer = nn.Linear(48,8)
    self.scalingfactor = (1/8)**(0.5)
    self.fclayer = nn.Linear(8,48)

  def forward(self, x):
    key = self.keyLayer(x)
    query = self.QueryLayer(x)
    value = self.ValueLayer(x)

    query_key = torch.matmul(query.transpose(-1, -2), key) * self.scalingfactor
    am = F.log_softmax(query_key, dim=-1)
    Z = torch.matmul(value, am)

    out = self.fclayer(Z)
    return out

class net(nn.Module):
  '''
  Add 3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)
  Apply GAP and get 1x1x48, call this X
  Repeat the Ultimus block 4 times
  Then add final FC layer that converts 48 to 10 and sends it to the loss function.
  Model would look like this C>C>C>U>U>U>U>FFC>Loss
  '''
  def __init__(self):
    super().__init__()
    self.convblock1 = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1),
        nn.RELU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3),
        nn.RELU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 48, 3, 1),
        nn.RELU(),
        nn.BatchNorm2d(48),
    )
      self.Gap = nn.AdaptiveAvgPool2d((1, 1))
      self.U1 = UltimusBlock()
      self.U2 = UltimusBlock()
      self.U3 = UltimusBlock()
      self.U4 = UltimusBlock()
      self.FC = nn.Linear(48,10)

    def forward(self, x):
        out = self.convblock1(x)
        out = self.Gap(out)
        out = out.view(out.size(0), -1)
        out = self.U1(out)
        out = self.U1(out)
        out = self.U1(out)
        out = self.U1(out)
        out = self.FC(out)
        return out
