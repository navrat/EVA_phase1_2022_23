## S8 Assignment: Building a custom resnet with one cycle learning policy on CIFAR & Application of GRADCAM
- The code base runs the custom RESNET18 on the CIFAR10 dataset.
- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
- There are 50000 training images and 10000 test images.
- The predictions are then evaluated visually using GRADCAM to understand the activation maps of the model.

## Proposed Architecture
- Add 3 Convolutions to arrive at AxAx48 dimensions (e.g. 32x32x3 | 3x3x3x16 >> 3x3x16x32 >> 3x3x32x48)
- Apply GAP and get 1x1x48, call this X
- Create a block called ULTIMUS that:
- Creates 3 FC layers called K, Q and V such that:
- X*K = 48*48x8 > 8
- X*Q = 48*48x8 > 8 
- X*V = 48*48x8 > 8 
- then create AM = SoftMax(QTK)/(8^0.5) = 8*8 = 8
- then Z = V*AM = 8*8 > 8
- then another FC layer called Out that:
- Z*Out = 8*8x48 > 48
- Repeat this Ultimus block 4 times
- Then add final FC layer that converts 48 to 10 and sends it to the loss function.
- Model would look like this C>C>C>U>U>U>U>FFC>Loss

## Results
The total number of parameters in the model were 26,074.

## Average loss after 24 epochs:

Training: Average loss: nan, Accuracy:  (10.00%)
Testing: Average loss: nan, Accuracy:  (10.00%)
class wise performance on Test Data:

The training log alongside epoch wise validation stats and the output of torchsummary can be referenced from the notebook.

## Training and Validation Performance Charts
image

## Misclassified Images
image
