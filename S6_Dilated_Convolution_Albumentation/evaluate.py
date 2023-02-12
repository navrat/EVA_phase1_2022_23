# single evaluation function to provide the entire classification model performance against given dataset 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def model_eval_f(model, device, test_loader, target_classes):
  '''
  - provides accuracy and loss for the dataset created via test_loader
  - provides a classwise accuracy table and classwise confusion matrix
  '''
  model.eval()

  train_losses = []
  test_losses = []
  train_acc = []
  test_acc = []
  test_loss = 0
  preds = []
  targets = []

  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          # print(data.shape)
          output = model(data)
          test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()
          incorrects = ~pred.eq(target.view_as(pred))
          preds.append(pred)
          targets.append(target)
  preds = [elem.squeeze(dim=1).cpu().numpy() for elem in preds]
  preds  = [element for sublist in preds for element in sublist]
  targets = [element for sublist in targets for element in sublist]
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
    
  print('\n class wise performance: \n')
  conf_matrix_all = confusion_matrix([target_classes[i] for i in targets], [target_classes[i] for i in preds])
  print(pd.DataFrame({'classes':target_classes,'accuracy': 100*conf_matrix_all.diagonal()/conf_matrix_all.sum(axis=1)}))
  print(classification_report([target_classes[i] for i in targets], [target_classes[i] for i in preds]))
