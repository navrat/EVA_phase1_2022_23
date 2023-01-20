Dataset: MNIST

Objective: digit classification

New target is:
- 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 10000 Parameters (additional points for doing this in less than 8000 pts)
- Do this in exactly 3 steps (or more)
- Each File must have "target, result, analysis" TEXT block (either at the start or the end)



**Solution:**

_Code 1: Building the skeleton_

Target:
- Create a light skeleton

Results:
- Parameters: 16.3k
- Best Train Accuracy: 98.81 (till 15 epochs)
- Best Test Accuracy: 98.74 (@ epoch ; till 15 epochs)

Analysis:
- Decent model with almost no overfitting
- Model does not peak by 15 epochs needs more capacity
- No epochs touch 99+ accuracy so far.


_Code 2: Adding batch normalization_

Target:
- Add Batch normalization
- Reduce kernel channels in the middle CNN layers

Results:
- Parameters: 14.8k
- Best Train Accuracy: 99.83 (till 15 epochs)
- Best Test Accuracy: 99.25 (@ epoch 15 ; till 15 epochs)

Analysis:
- Model has overfit considerably.
- Train and test accuracies have both improved compared to initial skeleton code. Train accuracy has almost maxed out.
- Consistency in test accuracies in later epochs of 99.25 - 99.27
- Adding dropouts to reduce this gap would be a next step
- Adding augmentation to generalize training would be a next step.


_Code 3: Adding dropout and augmentation_

Target:
- dropout on each layer
- GAP with a conv layer to reduce number of params
- Create more features in the inital layers using an expand-squeeze approch

Results:
- Parameters: 11.4k
- Best Train Accuracy: 99.14 (till 15 epochs)
- Best Test Accuracy: 99.38 (@ multiple epochs ; till 15 epochs)

Analysis:
- Model has not overfit and can further improve.
- Train and test accuracies have both improved compared to batch-norm code.
- Consistency in test accuracies in later epochs of 99.29 - 99.38
- Adding data augmentation would be a next step
- Adding LR optimizer would be a next step


_Code 4: Adding GAP and LR scheduler_

Target:
- Add rotation based augmentation
- Add LR Scheduler
- Reduce params from conv5 and bring the total param to 10k

Results:
- Parameters: 10k
- Best Train Accuracy: 98.80 (till 15 epochs)
- Best Test Accuracy: 99.35 (@ last 3 epochs ; till 15 epochs)

Analysis:
- Model has not overfit due to multiple regularization approaches.
- Train and test accuracies rose sharply in early epochs when compared to code1, code2, code3. 
- Consistency in test accuracies in later epochs of 99.35
- Model can still be likely improved with lesser params and a better learning rate approach.
