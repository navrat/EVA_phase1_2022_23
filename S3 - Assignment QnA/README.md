Forward and BackPropagation Calculations at different Learning Rates for a sample input of a 3 layer neural network
- defining inputs, weights and target values
- feedforward calculations using sigmoid activations to get total error at each step (use of mean square error as loss)
- using chain rule, gradient calculations through the output layer first to get gradients for w5 to w8
- using chain rule, gradient calculations through the hidden layer (using the prior gradients of the weights between hidden and o/p layer) to get gradients for w1 to w5
- use of a learning rate to update weights based on direction of gradients to minimize loss
- run this for multiple iterations and track weight changes and loss reduction
<img width="960" alt="image" src="https://user-images.githubusercontent.com/31410799/212448434-98577c20-583b-42f7-a997-77c74d21d159.png">
