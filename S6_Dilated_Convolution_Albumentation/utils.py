# import necessary libraries
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

def unnormalize(tensor, mean=(0.4914, 0.4822, 0.4471), std=(0.2469, 0.2433, 0.2615)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
  
 def imshow(img):
    img = unnormalize(img)    # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def plot_sample_images(dataloader_obj, batch_size=5)
# get some random training images
dataiter = iter(dataloader_obj)
images, labels = next(dataiter)
# show images
imshow(torchvision.utils.make_grid(images[0:batch_size]))
# print labels
print([classes[i] for i in labels[0:4]])

def plot_mispredictions(model, device, test_loader, target_classes):
  '''
  - provides accuracy and loss for the dataset created via test_loader
  - provides a crosstab of all mispredictions to identify where model commonly fails
  - takes the first 10 mispredictions and plots them on a 5X2 grid
  '''
  model.eval()

  train_losses = []
  test_losses = []
  train_acc = []
  test_acc = []
  test_loss = 0
  correct = 0
  preds = []
  targets = []
  incorrect_pred_targets = []
  incorrect_preds = []
  incorrect_pred_images = []
  incorrect_pred_images_all = []

  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          # print(data.shape)
          output = model(data)
          test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()
          incorrects = ~pred.eq(target.view_as(pred))
          if len(target[incorrects.squeeze(dim=1)])>0:
            incorrect_pred_targets.append(target[incorrects.squeeze(dim=1)])
            incorrect_preds.append(pred[incorrects.squeeze(dim=1)])
            incorrect_pred_images.append(data[incorrects.squeeze(dim=1)])
          [element for sublist in a for element in sublist]
  incorrect_pred_targets_all = [t.cpu().numpy() for t in incorrect_pred_targets]
  incorrect_pred_targets_all = [element for sublist in incorrect_pred_targets_all for element in sublist]
  incorrect_preds_all = [t.cpu().numpy() for t in incorrect_preds]
  incorrect_preds_all = [element for sublist in incorrect_preds_all for element in sublist]
  incorrect_preds_all = [element for sublist in incorrect_preds_all for element in sublist]
  incorrect_pred_images_all = [element for sublist in incorrect_pred_images for element in sublist]

  print("\n # of incorrect images predicted in test dataset of {}: {}".format(len(test_loader.dataset), len(incorrect_pred_images_all)))
  figure = plt.figure(figsize = (10,12))
  num_of_images = 10
  for index in range(1, num_of_images + 1):
      plt.subplot(5, 2, index)
      # plt.axis('off')
#       print(incorrect_pred_images_all[index].cpu().numpy().shape)
      plt.imshow(unnormalize(incorrect_pred_images_all[index].cpu()))
      plt.title(f'Target- {target_classes[incorrect_pred_targets_all[index]]} ; Predicted- {target_classes[incorrect_preds_all[index]]}')

  print("\n\n crosstab of incorrect prediction to understand patterns: \n")
  return pd.crosstab(pd.Series([target_classes[i] for i in incorrect_pred_targets_all], name='Actual'), pd.Series([target_classes[i] for i in incorrect_preds_all], name='Predicted'))
