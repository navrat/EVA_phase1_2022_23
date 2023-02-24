## S8 Assignment: Building a custom resnet with one cycle learning policy on CIFAR & Application of GRADCAM
- The code base runs the custom RESNET18 on the CIFAR10 dataset.
- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
- There are 50000 training images and 10000 test images.
- The predictions are then evaluated visually using GRADCAM to understand the activation maps of the model.

### Proposed Architecture

- PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
- Layer1 -
  - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
  - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
  - Add(X, R1)
- Layer 2 -
  - Conv 3x3 [256k]
  - MaxPooling2D
  - BN
  - ReLU
- Layer 3 -
  - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
  - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
  - Add(X, R2)
- MaxPooling with Kernel Size 4
- FC Layer 
- SoftMax
- Uses One Cycle Policy such that:
  - Total Epochs = 24
  - Max at Epoch = 5
  - LRMIN = FIND
  - LRMAX = FIND
  - NO Annihilation
- Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
- Batch size = 512
- Target Accuracy: 90% (93.8% quadruple scores). 

## Results
- The total number of parameters in the model were xx to achieve required accuracy.
- Average loss after 20 epochs:
  - Training: Loss = ?? ; Accuracy = ??
  - Testing: Loss = ?? ; Accuracy: ??

- class wise performance on Test Data:

 
- The training log alongside epoch wise validation stats and the output of torchsummary can be referenced from the notebook.

## Misclassified Images


## GradCam of Misclassified Images

