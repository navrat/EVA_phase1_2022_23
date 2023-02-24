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
  - Training: Average loss: 0.1717, Accuracy: 47014/50000 (94.03%)
  - Testing: Average loss: 0.2693, Accuracy: 9121/10000 (91.21%)

- class wise performance on Test Data:
<img width="110" alt="image" src="https://user-images.githubusercontent.com/31410799/221163741-f450efea-5ad7-4d4d-80f1-3306cdbbe308.png">
 
- The training log alongside epoch wise validation stats and the output of torchsummary can be referenced from the notebook.

## Misclassified Images

![image](https://user-images.githubusercontent.com/31410799/221162933-7cb988fc-824f-4739-889b-dab576eba7ce.png)

## GradCam of Misclassified Images

- Layer3:

![image](https://user-images.githubusercontent.com/31410799/221163016-d2fbdefe-5150-48e1-8a94-b89780ef234a.png)


- Layer2:
![image](https://user-images.githubusercontent.com/31410799/221163140-0759f9f7-9ca5-401f-90f3-d0fcaf3dacff.png)

