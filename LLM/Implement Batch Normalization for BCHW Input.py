"""
Implement a function that performs Batch Normalization on a 4D NumPy array representing a batch of feature maps in the BCHW format (batch, channels, height, width). The function should normalize the input across the batch and spatial dimensions for each channel, then apply scale (gamma) and shift (beta) parameters. Use the provided epsilon value to ensure numerical stability.

Example:
Input:
B, C, H, W = 2, 2, 2, 2; np.random.seed(42); X = np.random.randn(B, C, H, W); gamma = np.ones(C).reshape(1, C, 1, 1); beta = np.zeros(C).reshape(1, C, 1, 1)
Output:
[[[[ 0.42859934, -0.51776438], [ 0.65360963,  1.95820707]], [[ 0.02353721,  0.02355215], [ 1.67355207,  0.93490043]]], [[[-1.01139563,  0.49692747], [-1.00236882, -1.00581468]], [[ 0.45676349, -1.50433085], [-1.33293647, -0.27503802]]]]
Reasoning:
The input X is a 2x2x2x2 array. For each channel, compute the mean and variance across the batch (B), height (H), and width (W) dimensions. Normalize X using (X - mean) / sqrt(variance + epsilon), then scale by gamma and shift by beta. The output matches the expected normalized values.

Understanding Batch Normalization
Batch Normalization (BN) is a widely used technique that helps to accelerate the training of deep neural networks and improve model performance. By normalizing the inputs to each layer so that they have a mean of zero and a variance of one, BN stabilizes the learning process, speeds up convergence, and introduces regularization, which can reduce the need for other forms of regularization like dropout.

Concepts
Batch Normalization operates on the principle of reducing internal covariate shift, which occurs when the distribution of inputs to a layer changes during training as the model weights get updated. This can slow down training and make hyperparameter tuning more challenging. By normalizing the inputs, BN reduces this problem, allowing the model to train faster and more reliably.

The process of Batch Normalization consists of the following steps:

Compute the Mean and Variance: For each mini-batch, compute the mean and variance of the activations for each feature (dimension).
Normalize the Inputs: Normalize the activations using the computed mean and variance.
Apply Scale and Shift: After normalization, apply a learned scale (gamma) and shift (beta) to restore the model's ability to represent the data's original distribution.
Training and Inference: During training, the mean and variance are computed from the current mini-batch. During inference, a running average of the statistics from the training phase is used.
Structure of Batch Normalization for BCHW Input
For an input tensor with the shape BCHW, where:

B: batch size,
C: number of channels,
H: height,
W: width, the Batch Normalization process operates on specific dimensions based on the task's requirement.
1. Mean and Variance Calculation
In Batch Normalization, we typically normalize the activations across the batch and over the spatial dimensions (height and width) for each channel. This means we calculate the mean and variance per channel (C) for the batch and spatial dimensions (H, W).
For each channel 
c
c, we compute the mean 
μ
c
μ 
c
​
  and variance 
σ
c
2
σ 
c
2
​
  over the mini-batch and spatial dimensions:

μ
c
=
1
B
⋅
H
⋅
W
∑
i
=
1
B
∑
h
=
1
H
∑
w
=
1
W
x
i
,
c
,
h
,
w
μ 
c
​
 = 
B⋅H⋅W
1
​
  
i=1
∑
B
​
  
h=1
∑
H
​
  
w=1
∑
W
​
 x 
i,c,h,w
​
 
σ
c
2
=
1
B
⋅
H
⋅
W
∑
i
=
1
B
∑
h
=
1
H
∑
w
=
1
W
(
x
i
,
c
,
h
,
w
−
μ
c
)
2
σ 
c
2
​
 = 
B⋅H⋅W
1
​
  
i=1
∑
B
​
  
h=1
∑
H
​
  
w=1
∑
W
​
 (x 
i,c,h,w
​
 −μ 
c
​
 ) 
2
 
Where:

x
i
,
c
,
h
,
w
x 
i,c,h,w
​
  is the input activation at batch index 
i
i, channel 
c
c, height 
h
h, and width 
w
w.
B
B is the batch size.
H
H and 
W
W are the spatial dimensions (height and width).
C
C is the number of channels.
The mean and variance are computed over all spatial positions (H, W) and across all samples in the batch (B) for each channel (C).

2. Normalization
Once the mean 
μ
c
μ 
c
​
  and variance 
σ
c
2
σ 
c
2
​
  have been computed for each channel, the next step is to normalize the input. The normalization is done by subtracting the mean and dividing by the standard deviation (plus a small constant 
ϵ
ϵ for numerical stability):

x
^
i
,
c
,
h
,
w
=
x
i
,
c
,
h
,
w
−
μ
c
σ
c
2
+
ϵ
x
^
  
i,c,h,w
​
 = 
σ 
c
2
​
 +ϵ
​
 
x 
i,c,h,w
​
 −μ 
c
​
 
​
 
Where:

x
^
i
,
c
,
h
,
w
x
^
  
i,c,h,w
​
  is the normalized activation for the input at batch index 
i
i, channel 
c
c, height 
h
h, and width 
w
w.
ϵ
ϵ is a small constant to avoid division by zero (for numerical stability).
3. Scale and Shift
After normalization, the next step is to apply a scale (
γ
c
γ 
c
​
 ) and shift (
β
c
β 
c
​
 ) to the normalized activations for each channel. These learned parameters allow the model to adjust the output distribution of each feature, preserving the flexibility of the original activations.

y
i
,
c
,
h
,
w
=
γ
c
x
^
i
,
c
,
h
,
w
+
β
c
y 
i,c,h,w
​
 =γ 
c
​
  
x
^
  
i,c,h,w
​
 +β 
c
​
 
Where:

γ
c
γ 
c
​
  is the scaling factor for channel 
c
c.
β
c
β 
c
​
  is the shifting factor for channel 
c
c.
4. Training and Inference
During Training: The mean and variance are computed for each mini-batch and used for normalization across the batch and spatial dimensions for each channel.
During Inference: The model uses a running average of the statistics (mean and variance) that were computed during training to ensure consistent behavior when deployed.
Key Points
Normalization Across Batch and Spatial Dimensions: In Batch Normalization for BCHW input, the normalization is done across the batch (B) and spatial dimensions (H, W) for each channel (C). This ensures that each feature channel has zero mean and unit variance, making the training process more stable.

Channel-wise Normalization: Batch Normalization normalizes the activations independently for each channel (C) because different channels in convolutional layers often have different distributions and should be treated separately.

Numerical Stability: The small constant 
ϵ
ϵ is added to the variance to avoid numerical instability when dividing by the square root of variance, especially when the variance is very small.

Improved Gradient Flow: By reducing internal covariate shift, Batch Normalization allows the gradients to flow more easily during backpropagation, helping the model train faster and converge more reliably.

Regularization Effect: Batch Normalization introduces noise into the training process because it relies on the statistics of a mini-batch. This noise acts as a form of regularization, which can prevent overfitting and improve generalization.

Why Normalize Over Batch and Spatial Dimensions?
Across Batch: Normalizing across the batch helps to stabilize the input distribution across all samples in a mini-batch. This allows the model to avoid the problem of large fluctuations in the input distribution as weights are updated.

Across Spatial Dimensions: In convolutional networks, the spatial dimensions (height and width) are highly correlated, and normalizing over these dimensions ensures that the activations are distributed consistently throughout the spatial field, helping to maintain a stable learning process.

Channel-wise Normalization: Each channel can have its own distribution of values, and normalization per channel ensures that each feature map is scaled and shifted independently, allowing the model to learn representations that are not overly sensitive to specific channels' scaling.

By normalizing across the batch and spatial dimensions and applying a per-channel transformation, Batch Normalization helps reduce internal covariate shift and speeds up training, leading to faster convergence and better overall model performance.
"""

import numpy as np

def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	# Your code here
	mean = X.mean(axis=(0,2,3),keepdims=True)
    variance = X.var(axis=(0,2,3),keepdims=True)
    X_norm = (X -  mean)/ np.sqrt(variance + epsilon)
    X_trans = gamma * X_norm + beta
    return X_trans
