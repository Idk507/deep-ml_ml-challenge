"""
Implement a dropout layer that applies random neuron deactivation during training to prevent overfitting in neural networks. The layer should randomly zero out a proportion of input elements based on a dropout rate p, scale the remaining values by 1/(1-p) to maintain expected values, and pass inputs unchanged during inference. During backpropagation, gradients must be masked with the same dropout pattern and scaled by the same factor to ensure proper gradient flow.

Example:
Input:
x = np.array([1.0, 2.0, 3.0, 4.0]), grad = np.array([0.1, 0.2, 0.3, 0.4]), p = 0.5
Output:
output = array([[2., 0. , 6. , 0. ]]), grad = array([[0.2, 0. , 0.6, 0. ]])
Reasoning:
The Dropout layer randomly zeroes out elements of the input tensor with probability p during training. To maintain the expected value of the activations, the remaining elements are scaled by a factor of 1 / (1 - p). During inference, Dropout is disabled and the input is passed through unchanged. During backpropagation, the same dropout mask and scaling are applied to the gradients, ensuring the expected gradient magnitude is preserved.


Implementing Dropout Layer
Introduction
Dropout is a regularization technique that randomly deactivates neurons during training to prevent overfitting. It forces the network to learn with different neurons and prevents it from becoming too dependent on specific neurons.

Learning Objectives
Understand the concept and purpose of dropout
Learn how dropout works during training and inference
Implement dropout layer with proper scaling
Theory
During training, dropout randomly sets a proportion of inputs to zero and scales up the remaining values to maintain the expected value. The mathematical formulation is:

During training:

y
=
x
⊙
m
1
−
p
y= 
1−p
x⊙m
​
 

During inference:

y
=
x
y=x

During backpropagation:

g
r
a
d
=
g
r
a
d
⊙
m
1
−
p
grad= 
1−p
grad⊙m
​
 

Where:

x
x is the input vector
m
m is a binary mask vector sampled from Bernoulli(p)
⊙
⊙ represents element-wise multiplication
p
p is the dropout rate (probability of keeping a neuron)
The mask 
m
m is randomly generated for each forward pass during training and is stored in memory to be used in the corresponding backward pass. This ensures that the same neurons are dropped during both forward and backward propagation for a given input.

The scaling factor 
1
1
−
p
1−p
1
​
  during training ensures that the expected value of the output matches the input, making the network's behavior consistent between training and inference.

During backpropagation, the gradients must also be scaled by the same factor 
1
1
−
p
1−p
1
​
  to maintain the correct gradient flow.

Dropout acts as a form of regularization by:

Preventing co-adaptation of neurons, forcing them to learn more robust features that are useful in combination with many different random subsets of other neurons
Creating an implicit ensemble of networks, as each forward pass uses a different subset of neurons, effectively training multiple networks that share parameters
Reducing the effective capacity of the network during training, which helps prevent overfitting by making the model less likely to memorize the training data
Read more at:

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15(1), 1929-1958. PDF
Problem Statement
Implement a dropout layer class that can be used during both training and inference phases of a neural network. The implementation should:

Apply dropout during training by randomly zeroing out elements
Scale the remaining values appropriately to maintain expected values
Pass through inputs unchanged during inference
Support backpropagation by storing and using the dropout mask
Requirements
The DropoutLayer class should implement:

__init__(p: float): Initialize with dropout probability p
forward(x: np.ndarray, training: bool = True) -> np.ndarray: Apply dropout during forward pass
backward(grad: np.ndarray) -> np.ndarray: Handle gradient flow during backpropagation
Input Parameters
p: Dropout rate (probability of keeping a neuron), must be between 0 and 1
x: Input tensor of any shape
training: Boolean flag indicating if in training mode
grad: Gradient tensor during backpropagation
Output
Forward pass: Tensor of same shape as input with dropout applied
Backward pass: Gradient tensor with dropout mask applied
Example
# Example usage:
x = np.array([1.0, 2.0, 3.0, 4.0])
grad = np.array([0.1, 0.2, 0.3, 0.4])
p = 0.5  # 50% dropout rate

# During training
output_train = dropout_layer(x, p, training=True)

# During inference
output_inference = dropout_layer(x, p, training=False)

# Backward
grad_back = dropout.backward(grad)
Tips
Use numpy's random binomial generator for creating the mask
Remember to scale up the output and gradients during training by 1/(1-p)
Test with different dropout rates (typically between 0.2 and 0.5)
Verify that the expected value of the output matches the input
Common Pitfalls
Using the same mask for all examples in a batch
Setting dropout rate too high (can lead to underfitting)


"""
import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer."""
        if p < 0 or p >= 1:
            raise ValueError("Dropout rate must be between 0 and 1 (1-exclusive)")

        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer."""
        if not training:
            return x

        self.mask = np.random.binomial(1, 1 - self.p, x.shape)

        return x * self.mask / (1 - self.p)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer."""
        if self.mask is None:
            raise ValueError("Forward pass must be called before backward pass")

        return grad * self.mask / (1 - self.p)

