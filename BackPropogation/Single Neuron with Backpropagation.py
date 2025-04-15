"""
Single Neuron with Backpropagation

Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. The function should update the weights and bias using gradient descent based on the MSE loss, and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.

Example:
Input:
features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], initial_weights = [0.1, -0.2], initial_bias = 0.0, learning_rate = 0.1, epochs = 2
Output:
updated_weights = [0.1036, -0.1425], updated_bias = -0.0167, mse_values = [0.3033, 0.2942]
Reasoning:
The neuron receives feature vectors and computes predictions using the sigmoid activation. Based on the predictions and true labels, the gradients of MSE loss with respect to weights and bias are computed and used to update the model parameters across epochs.

"""
"""
Neural Network Learning with Backpropagation
This task involves implementing backpropagation for a single neuron in a neural network. The neuron processes inputs and updates parameters to minimize the Mean Squared Error (MSE) between predicted outputs and true labels.

Mathematical Background
Forward Pass
Compute the neuron output by calculating the dot product of the weights and input features, and adding the bias:

z
=
w
1
x
1
+
w
2
x
2
+
⋯
+
w
n
x
n
+
b
z=w 
1
​
 x 
1
​
 +w 
2
​
 x 
2
​
 +⋯+w 
n
​
 x 
n
​
 +b
σ
(
z
)
=
1
1
+
e
−
z
σ(z)= 
1+e 
−z
 
1
​
 
Loss Calculation (MSE)
The Mean Squared Error quantifies the error between the neuron's predictions and the actual labels:

M
S
E
=
1
n
∑
i
=
1
n
(
σ
(
z
i
)
−
y
i
)
2
MSE= 
n
1
​
  
i=1
∑
n
​
 (σ(z 
i
​
 )−y 
i
​
 ) 
2
 
Backward Pass (Gradient Calculation)
Compute the gradient of the MSE with respect to each weight and the bias. This involves the partial derivatives of the loss function with respect to the output of the neuron, multiplied by the derivative of the sigmoid function:

∂
M
S
E
∂
w
j
=
2
n
∑
i
=
1
n
(
σ
(
z
i
)
−
y
i
)
σ
′
(
z
i
)
x
i
j
∂w 
j
​
 
∂MSE
​
 = 
n
2
​
  
i=1
∑
n
​
 (σ(z 
i
​
 )−y 
i
​
 )σ 
′
 (z 
i
​
 )x 
ij
​
 
∂
M
S
E
∂
b
=
2
n
∑
i
=
1
n
(
σ
(
z
i
)
−
y
i
)
σ
′
(
z
i
)
∂b
∂MSE
​
 = 
n
2
​
  
i=1
∑
n
​
 (σ(z 
i
​
 )−y 
i
​
 )σ 
′
 (z 
i
​
 )
Parameter Update
Update each weight and the bias by subtracting a portion of the gradient, determined by the learning rate:

w
j
=
w
j
−
α
∂
M
S
E
∂
w
j
w 
j
​
 =w 
j
​
 −α 
∂w 
j
​
 
∂MSE
​
 
b
=
b
−
α
∂
M
S
E
∂
b
b=b−α 
∂b
∂MSE
​
 
Practical Implementation
This process refines the neuron's ability to predict accurately by iteratively adjusting the weights and bias based on the error gradients, optimizing the neural network's performance over multiple iterations.


"""


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    weights = np.array(initial_weights)
    bias = initial_bias
    features = np.array(features)
    labels = np.array(labels)
    mse_values = []

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        predictions = sigmoid(z)
        
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        # Gradient calculation for weights and bias
        errors = predictions - labels
        weight_gradients = (2/len(labels)) * np.dot(features.T, errors * predictions * (1 - predictions))
        bias_gradient = (2/len(labels)) * np.sum(errors * predictions * (1 - predictions))
        
        # Update weights and bias
        weights -= learning_rate * weight_gradients
        bias -= learning_rate * bias_gradient

        # Round weights and bias for output
        updated_weights = np.round(weights, 4)
        updated_bias = round(bias, 4)

    return updated_weights.tolist(), updated_bias, mse_values
