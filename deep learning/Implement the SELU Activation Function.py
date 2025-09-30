"""
Implement the SELU Activation Function
Implement the SELU (Scaled Exponential Linear Unit) activation function, a self-normalizing variant of ELU. Your task is to compute the SELU value for a given input while ensuring numerical stability.

Example:
Input:
selu(-1.0)
Output:
-1.1113
Reasoning:
For x = -1.0, the SELU activation is calculated using the formula 
S
E
L
U
(
x
)
=
λ
α
(
e
x
−
1
)
SELU(x)=λα(e 
x
 −1). Substituting the values of 
λ
λ and 
α
α, we get 
S
E
L
U
(
−
1.0
)
=
1.0507
×
1.6733
×
(
e
−
1.0
−
1
)
=
−
1.1113
SELU(−1.0)=1.0507×1.6733×(e 
−1.0
 −1)=−1.1113.

 Understanding the SELU Activation Function
The SELU (Scaled Exponential Linear Unit) activation function is a self-normalizing variant of the ELU activation function, introduced in 2017. It's particularly useful in deep neural networks as it automatically ensures normalized outputs with zero mean and unit variance.

Mathematical Definition
The SELU function is defined as:

S
E
L
U
(
x
)
=
λ
{
x
if 
x
>
0
α
(
e
x
−
1
)
if 
x
≤
0
SELU(x)=λ{ 
x
α(e 
x
 −1)
​
  
if x>0
if x≤0
​
 
Where:

λ
≈
1.0507
λ≈1.0507 is the scale parameter
α
≈
1.6733
α≈1.6733 is the alpha parameter
Characteristics
Output Range: The function maps inputs to 
(
−
λ
α
,
∞
)
(−λα,∞)
Self-Normalizing: Automatically maintains mean close to 0 and variance close to 1
Continuous: The function is continuous and differentiable everywhere
Non-Linear: Provides non-linearity while preserving gradients for negative values
Parameters: Uses carefully chosen values for 
λ
λ and 
α
α to ensure self-normalization
Advantages
Self-Normalization: Eliminates the need for batch normalization in many cases
Robust Learning: Helps prevent vanishing and exploding gradients
Better Performance: Often leads to faster training in deep neural networks
Internal Normalization: Maintains normalized activations throughout the network
Use Cases
SELU is particularly effective in:

Deep neural networks where maintaining normalized activations is crucial
Networks that require self-normalizing properties
Scenarios where batch normalization might be problematic or expensive

"""

import numpy as np
def selu(x: float) -> float:
	"""
	Implements the SELU (Scaled Exponential Linear Unit) activation function.

	Args:
		x: Input value

	Returns:
		SELU activation value
	"""
	alpha = 1.6732632423543772
	scale = 1.0507009873554804
    x = np.array(x)
	# Your code here
	if x > 0 : return scale * x
    else : return scale * (alpha*(np.exp(x)-1))


