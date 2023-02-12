## S7 Assignment: Use of RESNET18 on CIFAR & Application of GRADCAM 
- The code base runs the pre-defined RESNET18 on the CIFAR10 dataset. 
- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
- There are 50000 training images and 10000 test images.
- The predictions are then evaluated visually using GRADCAM to understand the activation maps of the model. 

### Architecture
- The original architecture here https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html
- GradCam implemention here https://github.com/jacobgil/pytorch-grad-cam


### Results
- The total number of parameters in the model were xx to achieve required accuracy. 
- Average loss after 20 epochs: 
  - Training: Loss = 0.71 ; Accuracy = 77.06%
  - Testing: Loss = 0.0041, Accuracy: 8200/10000 (82.00%)
  - class wise performance on Test Data: 

           classes  accuracy
         0   plane      73.8
         1     car      92.0
         2    bird      68.5
         3     cat      80.3
         4    deer      68.7
         5     dog      89.2
         6    frog      84.3
         7   horse      83.8
         8    ship      91.7
         9   truck      87.7
- The training log alongside epoch wise validation stats and the output of torchsummary can be referenced from the notebook.

### Misclassified Images
![image](https://user-images.githubusercontent.com/31410799/218307942-c4cc4fb6-376e-4259-81e2-4ae347dd7905.png)

## GradCam of misclassifeid Images

