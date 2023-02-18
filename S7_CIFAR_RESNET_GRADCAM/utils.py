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

import os
import sys
import time
import math
import torch.nn.init as init
import cv2

from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, ShiftScaleRotate, CoarseDropout, Cutout, PadIfNeeded
from albumentations.pytorch.transforms import ToTensorV2

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - unnormalize: reverse normalization of images
    - imshow: standard approach to printing an image
    - plot_sample_images: utility to plot use defined number of images from selected dataset
    - summary_statistics: to plot key aggregate statistics of any dataset
    - plot_mispredictions: to plot user defined number of incorrectly predicted images
    - model_eval_f: to plot overall and class wise performance measures for the model on selected dataset
'''

  
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

def plot_evolution_graph(trainObj, testObj):

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    # Loss Plot
    trainObj_losses_array = [i.cpu().item() for i in trainObj.train_losses]
    ax[0,0].plot(trainObj_losses_array)
    ax[0,0].set_xlabel('# batches')
    ax[0,0].set_ylabel('Loss')
    ax[0,0].set_title('Loss vs. # batches')
    ax[0,0].legend('training')
    
    testObj_losses_array = [i for i in testObj.test_losses]
    # ax2 = ax[0].twinx()
    ax[0,1].plot(testObj_losses_array)
    ax[0,1].set_xlabel('# Epochs')
    ax[0,1].set_ylabel('Loss')
    ax[0,1].set_title('Loss vs. # Epochs')
    ax[0,1].legend('test')

    # Accuracy Plot
    ax[1,0].plot(trainObj.train_acc, label='Training Accuracy')
    ax[1,0].set_xlabel('# batches')
    ax[1,0].set_ylabel('Accuracy')
    ax[1,0].set_title('Accuracy vs. # batches')
    ax[1,0].legend('training')

    ax[1,1].plot(testObj.test_acc, label='Training Accuracy')
    ax[1,1].set_xlabel('# Epochs')
    ax[1,1].set_ylabel('Accuracy')
    ax[1,1].set_title('TAccuracy vs. # Epochs')
    ax[1,1].legend('test')

    plt.tight_layout()
    plt.show()



    
'''GradCAM in PyTorch.
Grad-CAM implementation in Pytorch
Reference:
[1] https://github.com/vickyliin/gradcam_plus_plus-pytorch
[2] The paper authors torch implementation: https://github.com/ramprs/grad-cam
'''

layer_finders = {}


def register_layer_finder(model_type):
    def register(func):
        layer_finders[model_type] = func
        return func
    return register


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


@register_layer_finder('resnet')
def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class GradCAM:
    """Calculate GradCAM salinecy map.
    Args:
        input: input image with shape of (1, 3, H, W)
        class_idx (int): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
    Return:
        mask: saliency map of the same spatial dimension with input
        logit: model output
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        gradcam = GradCAM.from_config(model_type='resnet', arch=resnet, layer_name='layer4')
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    """

    def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module):
        self.model_arch = arch

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    @classmethod
    def from_config(cls, arch: torch.nn.Module, model_type: str, layer_name: str):
        target_layer = layer_finders[model_type](arch, layer_name)
        return cls(arch, target_layer)

    def saliency_map_size(self, *input_size):
        device = next(self.model_arch.parameters()).device
        self.model_arch(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

def plotGradCAM(net, testloader, classes, device, layer_name='layer4'):


    net.eval()

    misclassified_images = []
    actual_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    misclassified_images.append(data[i])
                    actual_labels.append(target[i])
                    predicted_labels.append(pred[i])

    gradcam = GradCAM.from_config(model_type='resnet', arch=net, layer_name=layer_name)

    fig = plt.figure(figsize=(10, 10))
    idx_cnt=1
    for idx in np.arange(10):

        img = misclassified_images[idx]
        lbl = predicted_labels[idx]
        lblp = actual_labels[idx]

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = img.unsqueeze(0).to(device)
        org_img = denormalize(img,mean=(0.4914, 0.4822, 0.4471),std=(0.2469, 0.2433, 0.2615))

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(img, class_idx=lbl)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, org_img, alpha=0.4)

        # Show images
        # for idx in np.arange(len(labels.numpy())):
        # Original picture
        
        ax = fig.add_subplot(5, 6, idx_cnt, xticks=[], yticks=[])
        npimg = np.transpose(org_img[0].cpu().numpy(),(1,2,0))
        ax.imshow(npimg, cmap='gray')
        ax.set_title(f"Label={str(classes[lblp])}\npred={classes[lbl]}")
        idx_cnt+=1

        ax = fig.add_subplot(5, 6, idx_cnt, xticks=[], yticks=[])
        npimg = np.transpose(heatmap,(1,2,0))
        ax.imshow(npimg, cmap='gray')
        ax.set_title("HeatMap".format(str(classes[lbl])))
        idx_cnt+=1

        ax = fig.add_subplot(5, 6, idx_cnt, xticks=[], yticks=[])
        npimg = np.transpose(cam_result,(1,2,0))
        ax.imshow(npimg, cmap='gray')
        ax.set_title("GradCAM".format(str(classes[lbl])))
        idx_cnt+=1

    fig.tight_layout()  
    plt.show()
