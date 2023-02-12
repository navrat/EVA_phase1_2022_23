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

def plot_sample_images(dataloader_obj, target_classes, batch_size=5):
    # get some random training images
    dataiter = iter(dataloader_obj)
    images, labels = next(dataiter)
    # show images
    imshow(torchvision.utils.make_grid(images[0:batch_size]))
    # print labels
    print([target_classes[i] for i in labels[0:batch_size]])

def data_summary_stats(dataset):
    # Load train data as numpy array
    train_data =dataset.data
    test_data = dataset.data

    total_data = np.concatenate((train_data, test_data), axis=0)
    print(total_data.shape)
    print(total_data.mean(axis=(0,1,2))/255)
    print(total_data.std(axis=(0,1,2))/255)
        


  

def plot_mispredictions(model, device, test_loader, target_classes, fig_size=(10,12), num_images=10):
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
  incorrect_pred_targets_all = [t.cpu().numpy() for t in incorrect_pred_targets]
  incorrect_pred_targets_all = [element for sublist in incorrect_pred_targets_all for element in sublist]
  incorrect_preds_all = [t.cpu().numpy() for t in incorrect_preds]
  incorrect_preds_all = [element for sublist in incorrect_preds_all for element in sublist]
  incorrect_preds_all = [element for sublist in incorrect_preds_all for element in sublist]
  incorrect_pred_images_all = [element for sublist in incorrect_pred_images for element in sublist]

  print("\n # of incorrect images predicted in test dataset of {}: {}".format(len(test_loader.dataset), len(incorrect_pred_images_all)))
  figure = plt.figure(figsize = fig_size)
  num_of_images = num_images
  for index in range(1, num_of_images + 1):
      plt.subplot(5, 2, index)
      # plt.axis('off')
      # print(incorrect_pred_images_all[index].cpu().numpy().shape)
      plt.imshow(unnormalize(incorrect_pred_images_all[index].permute(1,2,0).cpu()))
      plt.title(f'Target- {target_classes[incorrect_pred_targets_all[index]]} ; Predicted- {target_classes[incorrect_preds_all[index]]}')
  plt.show()
  print("\n\n crosstab of incorrect prediction to understand patterns: \n")
  return pd.crosstab(pd.Series([target_classes[i] for i in incorrect_pred_targets_all], name='Actual'), pd.Series([target_classes[i] for i in incorrect_preds_all], name='Predicted'))

# single evaluation function to provide the entire classification model performance against given dataset 
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
  correct = 0
  incoorects=0

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
