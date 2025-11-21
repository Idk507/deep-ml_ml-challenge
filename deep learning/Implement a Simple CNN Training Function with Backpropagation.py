"""
Create a function that trains a basic Convolutional Neural Network (CNN) using backpropagation. The network should include one convolutional layer with ReLU activation, followed by flattening and a dense layer with softmax output, and a cross entropy loss. You need to handle the forward pass, compute the loss gradients, and update the weights and biases using stochastic gradient descent. Ensure the function processes input data as grayscale images and one-hot encoded labels, and returns the trained weights and biases for the convolutional and dense layers.

Example:
Input:
import numpy as np; np.random.seed(42); X = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]); y = np.array([[1, 0]]); print(train_simple_cnn_with_backprop(X, y, 1, 0.01, 3, 1))
Output:
(array([[[ 0.00501739],        [-0.00128214],        [ 0.00662764]],       [[ 0.01543131],        [-0.00209028],        [-0.00203986]],       [[ 0.01614389],        [ 0.00807636],        [-0.00424248]]]), array([5.02517066e-05]), array([[ 0.00635715, -0.00556573]]), array([ 0.00499531, -0.00499531]))
Reasoning:
The solution processes the input X through a forward pass, where it applies a convolutional layer with ReLU activation, flattens the output, and passes it through a dense layer with softmax to compute predictions and loss based on the one-hot encoded label y. In the backward pass, it calculates gradients using backpropagation for the weights and biases, then updates them using stochastic gradient descent with the specified learning rate, and returns the updated weights after one epoch.


Understanding a Simple Convolutional Neural Network with Backpropagation
A Convolutional Neural Network (CNN) learns two things at once:

What to look for - small filters (kernels) that detect edges, textures, etc.
How to combine those detections - a dense layer that converts them into class probabilities.
Below is the full training loop broken into intuitive steps that can be implemented directly in NumPy.

1. Forward Pass
Convolution
The convolution layer slides a small filter over the input and produces feature maps:

Z
c
[
p
,
q
,
k
]
=
∑
i
,
j
X
[
p
+
i
,
q
+
j
]
⋅
W
c
[
i
,
j
,
k
]
+
b
c
[
k
]
Z 
c
 [p,q,k]= 
i,j
∑
​
 X[p+i,q+j]⋅W 
c
 [i,j,k]+b 
c
 [k]
This results in a tensor of shape 
(
H
−
k
+
1
,
W
−
k
+
1
,
F
)
(H−k+1,W−k+1,F), where 
H
H and 
W
W are the input height and width, 
k
k is the kernel size, and 
F
F is the number of filters.

ReLU Activation

A
c
=
max
⁡
(
0
,
Z
c
)
A 
c
 =max(0,Z 
c
 )
This introduces non-linearity by zeroing out negative values.

Flattening

The feature maps are reshaped into a vector:

A
f
=
flatten
(
A
c
)
A 
f
 =flatten(A 
c
 )
Dense Layer

Z
d
=
A
f
⋅
W
d
+
b
d
Z 
d
 =A 
f
 ⋅W 
d
 +b 
d
 
Each entry in 
A
f
A 
f
  contributes to every output class via weight matrix 
W
d
W 
d
  and bias 
b
d
b 
d
 .

Softmax Activation

y
^
c
=
e
Z
c
d
∑
j
e
Z
j
d
y
^
​
  
c
​
 = 
∑ 
j
​
 e 
Z 
j
d
​
 
 
e 
Z 
c
d
​
 
 
​
 
This converts raw scores into probabilities for classification.

2. Loss Function â Cross Entropy
For one-hot encoded label 
y
y and prediction 
y
^
y
^
​
 :

L
(
y
^
,
y
)
=
−
∑
c
y
c
log
⁡
(
y
^
c
)
L( 
y
^
​
 ,y)=− 
c
∑
​
 y 
c
​
 log( 
y
^
​
  
c
​
 )
This penalizes incorrect predictions based on confidence.

3. Backward Pass
Gradient of Softmax + Cross Entropy

∂
L
∂
Z
d
=
y
^
−
y
∂Z 
d
 
∂L
​
 = 
y
^
​
 −y
Dense Layer Gradients

∂
L
∂
W
d
=
(
A
f
)
T
⋅
∂
L
∂
Z
d
∂W 
d
 
∂L
​
 =(A 
f
 ) 
T
 ⋅ 
∂Z 
d
 
∂L
​
 , and the gradient with respect to biases is 
∂
L
∂
b
d
=
∂
L
∂
Z
d
∂b 
d
 
∂L
​
 = 
∂Z 
d
 
∂L
​
 .

Reshape the upstream gradient to the shape of 
A
c
A 
c
  for backpropagation through ReLU.

ReLU Gradient

∂
L
∂
Z
c
=
∂
L
∂
A
c
⋅
1
(
Z
c
>
0
)
∂Z 
c
 
∂L
​
 = 
∂A 
c
 
∂L
​
 ⋅1(Z 
c
 >0)
Convolution Filter Gradients

For each filter 
k
k:

∂
L
∂
W
i
,
j
,
k
c
=
∑
p
,
q
∂
L
∂
Z
p
,
q
,
k
c
⋅
X
p
+
i
,
q
+
j
∂W 
i,j,k
c
​
 
∂L
​
 = 
p,q
∑
​
  
∂Z 
p,q,k
c
​
 
∂L
​
 ⋅X 
p+i,q+j
​
 
∂
L
∂
b
k
c
=
∑
p
,
q
∂
L
∂
Z
p
,
q
,
k
c
∂b 
k
c
​
 
∂L
​
 = 
p,q
∑
​
  
∂Z 
p,q,k
c
​
 
∂L
​
 
4. Updating Parameters (SGD)
With learning rate 
η
η:

W
←
W
−
η
⋅
∂
L
∂
W
W←W−η⋅ 
∂W
∂L
​
 
b
←
b
−
η
⋅
∂
L
∂
b
b←b−η⋅ 
∂b
∂L
​
 
Repeat this process for each sample (stochastic gradient descent) and for multiple epochs.

5. Example Walkthrough
Suppose 
X
X is a grayscale image:

X
=
[
1
2
3
4
5
6
7
8
9
]
X= 
​
  
1
4
7
​
  
2
5
8
​
  
3
6
9
​
  
​
 
And the kernel is:

K
=
[
1
0
0
−
1
]
K=[ 
1
0
​
  
0
−1
​
 ]
Perform convolution at the top-left:

(
1
⋅
1
+
2
⋅
0
+
4
⋅
0
+
5
⋅
(
−
1
)
)
=
1
−
5
=
−
4
(1⋅1+2⋅0+4⋅0+5⋅(−1))=1−5=−4
After ReLU: max(0, -4) = 0
Flatten the result -> Dense layer -> Softmax output -> Compute loss

Backpropagate the error to adjust weights, and repeat to learn better filters and classifications over time.


"""
import numpy as np

def train_simple_cnn_with_backprop(X, y, epochs, learning_rate, kernel_size=3, num_filters=1):
    """
    Trains a simple CNN with:
    - One convolutional layer + ReLU
    - Flatten
    - Dense layer + softmax
    - Cross entropy loss
    - SGD updates
    
    Args:
        X: np.array of shape (H, W) or (N, H, W) grayscale images
        y: np.array of shape (N, C) one-hot labels
        epochs: int, number of epochs
        learning_rate: float, SGD step size
        kernel_size: int, size of convolution kernel
        num_filters: int, number of filters
    
    Returns:
        (conv_weights, conv_biases, dense_weights, dense_biases)
    """
    # Ensure batch dimension
    if X.ndim == 2:
        X = X[np.newaxis, ...]
    N, H, W = X.shape
    C = y.shape[1]  # number of classes
    
    # Initialize parameters
    conv_weights = np.random.randn(kernel_size, kernel_size, num_filters) * 0.01
    conv_biases = np.zeros(num_filters)
    
    conv_out_h = H - kernel_size + 1
    conv_out_w = W - kernel_size + 1
    dense_weights = np.random.randn(conv_out_h * conv_out_w * num_filters, C) * 0.01
    dense_biases = np.zeros(C)
    
    # Training loop
    for epoch in range(epochs):
        for n in range(N):
            x = X[n]
            target = y[n]
            
            # ----- Forward pass -----
            # Convolution
            Zc = np.zeros((conv_out_h, conv_out_w, num_filters))
            for f in range(num_filters):
                for i in range(conv_out_h):
                    for j in range(conv_out_w):
                        region = x[i:i+kernel_size, j:j+kernel_size]
                        Zc[i, j, f] = np.sum(region * conv_weights[:, :, f]) + conv_biases[f]
            Ac = np.maximum(0, Zc)  # ReLU
            
            # Flatten
            Af = Ac.flatten()
            
            # Dense
            Zd = Af @ dense_weights + dense_biases
            exp_scores = np.exp(Zd - np.max(Zd))
            y_hat = exp_scores / np.sum(exp_scores)
            
            # Loss (not used for update, but for gradient)
            # L = -np.sum(target * np.log(y_hat + 1e-9))
            
            # ----- Backward pass -----
            dZd = y_hat - target  # gradient wrt dense pre-activation
            dWd = np.outer(Af, dZd)
            dbd = dZd
            
            dAf = dense_weights @ dZd
            dAc = dAf.reshape(conv_out_h, conv_out_w, num_filters)
            dZc = dAc * (Zc > 0)  # ReLU backprop
            
            dWc = np.zeros_like(conv_weights)
            dbc = np.zeros_like(conv_biases)
            for f in range(num_filters):
                for i in range(conv_out_h):
                    for j in range(conv_out_w):
                        region = x[i:i+kernel_size, j:j+kernel_size]
                        dWc[:, :, f] += dZc[i, j, f] * region
                        dbc[f] += dZc[i, j, f]
            
            # ----- Update -----
            dense_weights -= learning_rate * dWd
            dense_biases -= learning_rate * dbd
            conv_weights -= learning_rate * dWc
            conv_biases -= learning_rate * dbc
    
    return conv_weights, conv_biases, dense_weights, dense_biases
