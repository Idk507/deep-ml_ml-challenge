"""
Implement the Local Response Normalization (LRN) operation introduced in the AlexNet paper. LRN applies a normalization across neighboring feature maps (channels) to encourage lateral inhibition mimicking a form of competition among neurons. Given a 4D tensor input of shape (N, C, H, W), your task is to normalize each activation using its neighbors within a local window along the channel dimension.

Example:
Input:
x = np.random.randn(1, 3, 2, 2); np.round(local_response_normalization(x, n=3, k=2, alpha=1e-4, beta=0.75), 4)
Output:
Example output shape: (1, 3, 2, 2)
Reasoning:
Each channel is normalized by a function of the sum of squared activations in its local neighborhood along the channel axis.

Learn About topic
Understanding Local Response Normalization (LRN)
Local Response Normalization (LRN) was introduced in the AlexNet paper (Krizhevsky et al., 2012) as a biologically inspired mechanism that encourages competition among neurons at the same spatial position but across neighboring channels.

Mathematical Definition
Given an input tensor 
a
x
,
y
i
a 
x,y
i
​
  at spatial location 
(
x
,
y
)
(x,y) and channel index 
i
i, the normalized output 
b
x
,
y
i
b 
x,y
i
​
  is defined as:

b
x
,
y
i
=
a
x
,
y
i
(
k
+
α
∑
j
=
m
a
x
(
0
,
i
−
n
/
2
)
m
i
n
(
N
−
1
,
i
+
n
/
2
)
(
a
x
,
y
j
)
2
)
β
b 
x,y
i
​
 = 
(k+α∑ 
j=max(0,i−n/2)
min(N−1,i+n/2)
​
 (a 
x,y
j
​
 ) 
2
 ) 
β
 
a 
x,y
i
​
 
​
 
Where:

n
n = local size (number of neighboring channels to normalize over)
k
k = additive constant (usually 2)
α
α = scaling parameter (e.g. 
10
−
4
10 
−4
 )
β
β = exponent parameter (e.g. 0.75)
Intuition
Competition: Neurons inhibit their neighbors' activations within a local region, improving generalization.
Contrast Enhancement: Encourages diversity in feature responses across channels.
Practical Effect: Often improves early convolutional layer performance.
Common Parameters
AlexNet used: 
n
=
5
,
k
=
2
,
α
=
10
−
4
,
β
=
0.75
n=5,k=2,α=10 
−4
 ,β=0.75

 """"

import numpy as np

def local_response_normalization(x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Applies Local Response Normalization (LRN) across the channel dimension.

    Args:
        x (np.ndarray): Input tensor of shape (N, C, H, W)
        n (int): Local window size (number of neighboring channels)
        k (float): Additive constant
        alpha (float): Scaling parameter
        beta (float): Exponent parameter

    Returns:
        np.ndarray: Normalized tensor of same shape as input
    """
    N, C, H, W = x.shape
    squared = x ** 2

    # Pad along channel dimension to handle borders
    pad = n // 2
    padded = np.pad(squared, ((0,0), (pad,pad), (0,0), (0,0)), mode='constant')

    # Compute normalization denominator
    scale = np.zeros_like(x)
    for i in range(C):
        # Sum over local window of size n
        scale[:, i, :, :] = np.sum(padded[:, i:i+n, :, :], axis=1)

    # Apply normalization
    denom = (k + alpha * scale) ** beta
    return x / denom
