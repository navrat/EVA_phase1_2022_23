'''Train CIFAR10 with PyTorch.'''
'''
Base Implementation referenced from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
Following additions made to the base implementation:
- training and test loops
- data selection and data split between test and train
- epochs
- batch size
- which optimizer to run
- do we run a scheduler?
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import resnet
import models

# from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# Source leveraged: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# import required libraries
import torch                  
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2

from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, ShiftScaleRotate, CoarseDropout
from albumentations.pytorch.transforms import ToTensorV2

class Compose_Train():
    def __init__(self):
        self.albumentations_transform = Compose([
            HorizontalFlip(),
            ShiftScaleRotate(),
            CoarseDropout (max_holes = 2, max_height=16, max_width=16, min_holes = 1, min_height=4, min_width=4, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], mask_fill_value = None),
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class Compose_Test():
    def __init__(self):
        self.albumentations_transform = Compose([
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])

    def __call__(self,img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img

class dataset_cifar10:
    def __init__(self, batch_size=128):
        # Defining CUDA
        cuda = torch.cuda.is_available()
        print("CUDA availability ?",cuda)

        # Defining data loaders with setting
        self.dataloaders_args = dict(shuffle=True, batch_size = batch_size, num_workers = 2, pin_memory = True) if cuda else dict(shuffle=True,batch_size = batch_size)
        self.sample_dataloaders_args = self.dataloaders_args.copy()

        self.classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def data(self, train_flag):

        # Training data augmentations
        if train_flag :
            return datasets.CIFAR10('./Data',
                            train=train_flag,
                            transform=Compose_Train(),
                            download=True)

        # Testing transformation - normalization
        else:
            return datasets.CIFAR10('./Data',
                                train=train_flag,
                                transform=Compose_Test(),
                                download=True)

    # Dataloader function
    def loader(self, train_flag=True):
        return(torch.utils.data.DataLoader(self.data(train_flag), **self.dataloaders_args))


    def data_summary_stats(self):
        # training_data 
        print("training data")
        train_data = self.data(train_flag=True).data
        print(train_data.mean(axis=(0,1,2))/255.)
        print(train_data.std(axis=(0,1,2))/255.)
        # testing_data 
        print("testing data")
        test_data = self.data(train_flag=False).data
        print(test_data.mean(axis=(0,1,2))/255.)
        print(test_data.std(axis=(0,1,2))/255.)
        # Load train data as numpy array
        print("total data")
        total_data = np.concatenate((train_data, test_data), axis=0)
        print(total_data.shape)
        print(total_data.mean(axis=(0,1,2))/255)
        print(total_data.std(axis=(0,1,2))/255)

# Model
def nnet_model():
  print('==> Building model..')
  # net = VGG('VGG19')
  net = resnet.ResNet18()
  # net = PreActResNet18()
  # net = GoogLeNet()
  # net = DenseNet121()
  # net = ResNeXt29_2x64d()
  # net = MobileNet()
  # net = MobileNetV2()
  # net = DPN92()
  # net = ShuffleNetG2()
  # net = SENet18()
  # net = ShuffleNetV2(1)
  # net = EfficientNetB0()
  # net = RegNetX_200MF()
  # net = SimpleDLA()
  net = net.to(device)
  if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True
  return net

class train:

    def __init__(self):

        self.train_losses = []
        self.train_acc    = []

    # Training
    def execute(self,net, device, trainloader, optimizer, criterion,epoch):

        #print('Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        #total = 0
        processed = 0
        pbar = tqdm(trainloader)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # get samples
            inputs, targets = inputs.to(device), targets.to(device)

            # Init
            optimizer.zero_grad()

            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            self.train_losses.append(loss.item())
            
            _, predicted = outputs.max(1)
            processed += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(desc= f'Epoch: {epoch},Loss={loss.item():3.2f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)

class test:

    def __init__(self):

        self.test_losses = []
        self.test_acc    = []

    def execute(self, net, device, testloader, criterion):

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(testloader.dataset)
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))

        # Save.
        self.test_acc.append(100. * correct / len(testloader.dataset))

def execute_run(net, device, trainloader, testloader, EPOCHS, lr=0.1):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    trainObj = train()
    testObj = test()

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        trainObj.execute(net, device, trainloader, optimizer, criterion, epoch)
        testObj.execute(net, device, testloader, criterion)
        scheduler.step()

    print('Finished Training')

    return trainObj, testObj

def model_training_setup(net, lr = 0.1, criterion="nll_loss", optimizer="sgd", scheduler = None, n_epochs = 20,):
  n_epochs = n_epochs
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  return criterion, optimizer, scheduler, n_epochs
  
'''
----------
Earlier training and testing definitions
----------
'''

from tqdm import tqdm
import torch

class train:

    def __init__(self):

        self.train_losses = []
        self.train_acc    = []

    # Training
    def execute(self,net, device, trainloader, optimizer, criterion,epoch):

        #print('Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        #total = 0
        processed = 0
        pbar = tqdm(trainloader)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # get samples
            inputs, targets = inputs.to(device), targets.to(device)

            # Init
            optimizer.zero_grad()

            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            _, predicted = outputs.max(1)
            processed += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(desc= f'Epoch: {epoch},Loss={loss.item():3.2f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)


class test:

    def __init__(self):

        self.test_losses = []
        self.test_acc    = []

    def execute(self, net, device, testloader, criterion):

        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(testloader.dataset)
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))

        # Save.
        self.test_acc.append(100. * correct / len(testloader.dataset))
