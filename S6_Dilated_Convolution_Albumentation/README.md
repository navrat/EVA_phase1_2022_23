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
- coarseDropout 

### Results
- The total number of parameters in the model were 200k to achieve required accuracy. 
- The training log alongside epoch wise validation stats and the output of torchsummary can be referenced from the notebook.
