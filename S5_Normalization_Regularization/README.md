# Last Modified: 29-Jan-2022
# Session 5 Assignment to cover the following:

## Leverage the 10k paramter code from S4 and build the following experiments:
Write a single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include.
Write a single notebook file to run all the 3 models above for 20 epochs each.

- Network with Group Normalization
- Network with Layer Normalization
- Network with L1 + BN

## Steps and calculations to perform the 3 normalizations techniques:
![image](https://user-images.githubusercontent.com/31410799/215351853-a3b5ca62-e79a-41fd-a7d6-4d93e0344dc6.png)

- Batch Normalization: BN tries to normalize the channels and computes mean and stddev along the direction of the batch. There are always as many number of mean and stddev as the number of output channels. The calculation of normalization takes place for every value in the output (but using the aggregates computed at a channel level).
- Layer Normalization: LN reverses this order of calculation and computes mean and stddev along the direction of the channels. There are always as many number of mean and stddev as the number of observations (i.e. batch size). The calculation of normalization still takes place for every value in the output (but using the aggregates computed for at an observation level).
- Group Normalization: GN takes the middle route where the mean and stddev are calculated along the direction of only a subset of channels. There are as many mean and stddev as the (number of groups of channels) X number of observations (i.e. batch size). The calculation of normalization still takes place for every value in the output (but using the aggregates computed for at an observation X channel_grouping level).

<img width="884" alt="image" src="https://user-images.githubusercontent.com/31410799/215353460-6446fda3-bbc7-43b6-b32b-85e057c5c41f.png">


## Findings from each of the 3 normalization techniques:

Findings from the 3 experiments:
- Effect of GN: Similar to the original BN from last experiment BUT not better than that.The misclassifications are low and relatively spread out (0, 1, 4, 5, 7)
- Effect of LN: Slightly lower but comparable performance with GN. The misclassifications are slightly more and concentrated on specific numbers (7, 5, 2)
- Effect of static L1_Lambda based L1Norm + BN: Considerably worse validation performance with current fixed L1_lambda strategy. Lots of mispredictions for 1, 4, 3, 7
- Of the 3 models, experiment with GN performs with the highest validation accuracy with very low degree of overfitting.
- Misclassifications are majorly same across the 3 models and are primarily centered on predicting off versions of 7, 4, 5; 

## Graphs highlighting accuracy and loss evolution over 20 epochs per experiment (4 charts: test/validation X loss/accuracy: 3 experiments per chart):
![image](https://user-images.githubusercontent.com/31410799/215352179-82f85dae-ca28-443e-9e6f-c27666974f82.png)


## Misclassified-images per experiment ( 3 collections in total):
Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images

- model 1 with GN:

![image](https://user-images.githubusercontent.com/31410799/215351816-dcd4d193-6b33-4651-8fee-a1b985f0c215.png)


- model 2 with LN:

![image](https://user-images.githubusercontent.com/31410799/215351806-33533fcd-1a66-48ab-83b9-1acbfecf66b1.png)


- model 3 with L1 + BN:

![image](https://user-images.githubusercontent.com/31410799/215351763-507374c8-0e7d-4423-b297-5f96098cddbb.png)




