## Forward and BackPropagation Calculations at different Learning Rates for a sample input of a 3 layer neural network
- defining inputs, weights and target values
- feedforward calculations using sigmoid activations to get total error at each step (use of mean square error as loss)
- using chain rule, gradient calculations through the output layer first to get gradients for w5 to w8
- using chain rule, gradient calculations through the hidden layer (using the prior gradients of the weights between hidden and o/p layer) to get gradients for w1 to w5
- use of a learning rate to update weights based on direction of gradients to minimize loss
- run this for multiple iterations and track weight changes and loss reduction
<img width="960" alt="image" src="https://user-images.githubusercontent.com/31410799/212448434-98577c20-583b-42f7-a997-77c74d21d159.png">


## Neural Network run with <20k parameters:

- Param List and architecture

Total params: 17,098

Trainable params: 17,098

Non-trainable params: 0


- Training and Validation Log:

  0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-28-83d789f1817a>:24: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)

loss=0.15329350531101227 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.94it/s]
Test set: Average loss: 0.0880, Accuracy: 9758/10000 (98%)

loss=0.014375776052474976 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.87it/s]
Test set: Average loss: 0.0723, Accuracy: 9788/10000 (98%)

loss=0.07859987020492554 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.64it/s]
Test set: Average loss: 0.0557, Accuracy: 9831/10000 (98%)

loss=0.07332415878772736 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.34it/s]
Test set: Average loss: 0.0387, Accuracy: 9890/10000 (99%)

loss=0.0197222251445055 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.68it/s]
Test set: Average loss: 0.0395, Accuracy: 9875/10000 (99%)

loss=0.005569449160248041 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.96it/s]
Test set: Average loss: 0.0337, Accuracy: 9904/10000 (99%)
