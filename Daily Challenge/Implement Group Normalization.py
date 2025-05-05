"""
Write a Python function to perform Group Normalization on a 4D input tensor with shape (B, C, H, W). The function should normalize over smaller groups of channels, then apply a learned scale (gamma) and shift (beta).
Example:
Input:
X.shape = (2, 2, 2, 2), gamma = [1, 1], beta = [0, 0], num_groups = 2
Output:
Normalized tensor where each group is independently normalized and scaled by gamma and shifted by beta.
Reasoning:
First split the channels into groups, compute mean and variance per group, normalize within the group, then scale and shift with gamma and beta.
Group Normalization (GN) is a normalization technique that divides the channels into groups and normalizes the activations within each group. Unlike Batch Normalization, which normalizes over the entire mini-batch, Group Normalization normalizes over groups of channels and is less dependent on the batch size. This makes it particularly useful for tasks with small batch sizes or when using architectures such as segmentation networks where spatial resolution is important.

Concepts
Group Normalization operates on the principle of normalizing within smaller groups of channels. The process reduces internal covariate shift within these groups and helps stabilize training, especially in scenarios where the batch size is small or varies across tasks.

The process of Group Normalization consists of the following steps:

Divide Channels into Groups: Split the feature channels into several groups. The number of groups is determined by the n_groups parameter.
Compute the Mean and Variance within Each Group: For each group, compute the mean and variance of the activations within the group, across the spatial dimensions and batch.
Normalize the Inputs: Normalize the activations of each group using the computed mean and variance.
Apply Scale and Shift: After normalization, apply a learned scale (gamma) and shift (beta) to restore the model's ability to represent the data's original distribution.
Structure of Group Normalization for BCHW Input
For an input tensor with the shape BCHW , where:

B: batch size,
C: number of channels,
H: height,
W: width, the Group Normalization process operates on specific dimensions based on the task's requirement.
1. Group Division
The input feature dimension C (channels) is divided into several groups. The number of groups is determined by the n_groups parameter, and the size of each group is calculated as:

groupSize
=
C
n
groups
groupSize= 
n 
groups
​
 
C
​
 
Where:

C is the number of channels.
n_groups is the number of groups into which the channels are divided.
groupSize is the number of channels in each group.
The input tensor is then reshaped to group the channels into the specified groups.

2. Mean and Variance Calculation within Groups
For each group, the mean 
μ
g
μ 
g
​
  and variance 
σ
g
2
σ 
g
2
​
  are computed over the spatial dimensions and across the batch. This normalization helps to stabilize the activations within each group.

μ
g
=
1
B
⋅
H
⋅
W
⋅
groupSize
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
∑
g
=
1
groupSize
x
i
,
g
,
h
,
w
μ 
g
​
 = 
B⋅H⋅W⋅groupSize
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
  
g=1
∑
groupSize
​
 x 
i,g,h,w
​
 
σ
g
2
=
1
B
⋅
H
⋅
W
⋅
groupSize
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
∑
g
=
1
groupSize
(
x
i
,
g
,
h
,
w
−
μ
g
)
2
σ 
g
2
​
 = 
B⋅H⋅W⋅groupSize
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
  
g=1
∑
groupSize
​
 (x 
i,g,h,w
​
 −μ 
g
​
 ) 
2
 
Where:

x
i
,
g
,
h
,
w
x 
i,g,h,w
​
  is the activation at batch index 
i
i, group index 
g
g, height 
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
groupSize
groupSize is the number of channels in each group.
3. Normalization
Once the mean 
μ
g
μ 
g
​
  and variance 
σ
g
2
σ 
g
2
​
  have been computed for each group, the next step is to normalize the input. The normalization is done by subtracting the mean and dividing by the standard deviation (square root of the variance, plus a small constant 
ϵ
ϵ for numerical stability):

x
^
i
,
g
,
h
,
w
=
x
i
,
g
,
h
,
w
−
μ
g
σ
g
2
+
ϵ
x
^
  
i,g,h,w
​
 = 
σ 
g
2
​
 +ϵ
​
 
x 
i,g,h,w
​
 −μ 
g
​
 
​
 
Where:

x
^
i
,
g
,
h
,
w
x
^
  
i,g,h,w
​
  is the normalized activation for the input at batch index 
i
i, group index 
g
g, height 
h
h, and width 
w
w.
ϵ
ϵ is a small constant to avoid division by zero.
4. Scale and Shift
After normalization, the next step is to apply a scale (
γ
g
γ 
g
​
 ) and shift (
β
g
β 
g
​
 ) to the normalized activations for each group. These learned parameters allow the model to adjust the output distribution of each group:

y
i
,
g
,
h
,
w
=
γ
g
x
^
i
,
g
,
h
,
w
+
β
g
y 
i,g,h,w
​
 =γ 
g
​
  
x
^
  
i,g,h,w
​
 +β 
g
​
 
Where:

γ
g
γ 
g
​
  is the scaling factor for group 
g
g.
β
g
β 
g
​
  is the shifting factor for group 
g
g.
5. Training and Inference
During Training: The mean and variance are computed for each mini-batch and used for normalization within each group.
During Inference: The model uses running averages of the statistics (mean and variance) that were computed during training to ensure consistent behavior when deployed.
Key Points
Group-wise Normalization: Group Normalization normalizes within smaller groups of channels instead of normalizing over the entire batch and all channels. This allows for more stable training in cases with small batch sizes.

Number of Groups: The number of groups is a hyperparameter (n_groups) that can significantly affect the modelâs performance. It is typically set to divide the total number of channels into groups of equal size.

Smaller Batch Sizes: Group Normalization is less dependent on the batch size, making it ideal for situations where batch sizes are small (e.g., segmentation tasks).

Numerical Stability: As with other normalization techniques, a small constant 
ϵ
ϵ is added to the variance to avoid numerical instability when dividing by the square root of variance.

Improved Convergence: Group Normalization can help improve the gradient flow throughout the network, making it easier to train deep networks with small batch sizes. It also helps speed up convergence and stabilize training.

Regularization Effect: Similar to Batch Normalization, Group Normalization introduces a form of regularization through the normalization process. It can reduce overfitting by acting as a noise source during training.

Why Normalize Over Groups?
Group-wise Normalization: By dividing the channels into smaller groups, Group Normalization ensures that each group has a stable distribution of activations, making it effective even when batch sizes are small.

Less Dependency on Batch Size: Unlike Batch Normalization, Group Normalization does not require large batch sizes to compute accurate statistics. This makes it well-suited for tasks such as image segmentation, where large batch sizes may not be feasible.

Channel-wise Learning: Group Normalization allows each group to learn independently, preserving flexibility while also controlling the complexity of normalization over channels.

By normalizing over smaller groups, Group Normalization can reduce internal covariate shift and allow for faster and more stable training, even in situations where Batch Normalization may be less effective due to small batch sizes.
"""


def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    '''
    Perform Group Normalization.
    
    Args:
    X: numpy array of shape (B, C, H, W), input data
    gamma: numpy array of shape (C,), scale parameter
    beta: numpy array of shape (C,), shift parameter
    num_groups: number of groups for normalization
    epsilon: small constant to avoid division by zero
    
    Returns:
    norm_X: numpy array of shape (B, C, H, W), normalized output
    '''
    batch_size, num_channels, height, width = X.shape
    group_size = num_channels // num_groups

    # Reshape X into groups
    X_reshaped = X.reshape(batch_size, num_groups, group_size, height, width)

    # Compute mean and variance for each group
    mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
    variance = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)

    X_norm = (X_reshaped - mean) / np.sqrt(variance + epsilon)

    # Reshape back to the original shape
    X_norm = X_norm.reshape(batch_size, num_channels, height, width)

    # Apply scale and shift
    norm_X = gamma * X_norm + beta
    return norm_X
