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

  classes  accuracy
0   plane      87.3
1     car      96.4
2    bird      81.7
3     cat      91.6
4    deer      84.8
5     dog      95.6
6    frog      93.3
7   horse      91.9
8    ship      95.3
9   truck      94.2
              precision    recall  f1-score   support

        bird       0.90      0.87      0.89      1000
         car       0.95      0.96      0.96      1000
         cat       0.82      0.82      0.82      1000
        deer       0.90      0.92      0.91      1000
         dog       0.88      0.85      0.86      1000
        frog       0.91      0.96      0.93      1000
       horse       0.95      0.93      0.94      1000
       plane       0.92      0.92      0.92      1000
        ship       0.96      0.95      0.96      1000
       truck       0.93      0.94      0.94      1000

    accuracy                           0.91     10000
   macro avg       0.91      0.91      0.91     10000
weighted avg       0.91      0.91      0.91     10000

- The training log alongside epoch wise validation stats and the output of torchsummary can be referenced from the notebook.

## Misclassified Images

![image](https://user-images.githubusercontent.com/31410799/221162933-7cb988fc-824f-4739-889b-dab576eba7ce.png)

## GradCam of Misclassified Images

- Layer3:

![image](https://user-images.githubusercontent.com/31410799/221163016-d2fbdefe-5150-48e1-8a94-b89780ef234a.png)


- Layer2:
![image](https://user-images.githubusercontent.com/31410799/221163140-0759f9f7-9ca5-401f-90f3-d0fcaf3dacff.png)

