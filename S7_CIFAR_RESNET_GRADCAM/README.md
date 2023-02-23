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
  - Testing: Loss = 0.0030, Accuracy: 8705/10000 (87.05%)
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
![image](https://user-images.githubusercontent.com/31410799/219889733-52fa4d9f-c722-4e7b-9183-5aa1259c12af.png)

![image](https://user-images.githubusercontent.com/31410799/219889877-2a96479f-8b43-47a5-872c-30f81e382314.png)


## GradCam of misclassifeid Images (Layer4)
![image](https://user-images.githubusercontent.com/31410799/220999066-cde33184-b59f-4d2d-b5b8-2daea8ed4d9a.png)

## GradCam of misclassifeid Images (Layer3)
![image](https://user-images.githubusercontent.com/31410799/221001967-f63c54c0-e1c7-4331-ad5b-fcb316c6acd6.png)

## GradCam of misclassifeid Images (Layer2)
![image](https://user-images.githubusercontent.com/31410799/221002075-b6ddde57-cc2a-4965-9b28-943863226f26.png)

## GradCam of misclassifeid Images (Layer1)
![image](https://user-images.githubusercontent.com/31410799/221002152-9d2970e8-1459-4a4a-bf6e-bba54f696817.png)
