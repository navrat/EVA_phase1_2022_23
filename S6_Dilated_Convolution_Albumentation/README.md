## S6 Assignment: Use of Dilated convolutions and albumentaitons
- The code base provides a build of a CNN using strided conv kernels and the Albumentations library on the CIFAR10 dataset. 
- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
- There are 50000 training images and 10000 test images.

### Architecture
The original architecture here https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw is changed to C1C2C3C40 architecture with following changes: 
- Replaced max pooling with 3 convolutional layers of 3x3 filters and a stride of 2. 
- The final layer utilizes global average pooling (GAP). 
- A dilated convolution layer.

### Augmentation Strategy
Three augmentations are performed using the albumentations library within the training data loader: 
- horizontal flipping
- shiftScaleRotate
- coarseDropout (max_holes = 2, max_height=16, max_width=16, min_holes = 1, min_height=4, min_width=4, fill_value=(mean of your dataset), mask_fill_value = None) 

### Results
- The total number of parameters in the model were 142570 (under 200k) to achieve required accuracy. 
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

