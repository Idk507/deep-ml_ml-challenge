"""
Write a Python function leaky_relu that implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function. The function should take a float z as input and an optional float alpha, with a default value of 0.01, as the slope for negative inputs. The function should return the value after applying the Leaky ReLU function.

Example:
Input:
print(leaky_relu(0)) 
print(leaky_relu(1))
print(leaky_relu(-1)) 
print(leaky_relu(-2, alpha=0.1))
Output:
0
1
-0.01
-0.2
Reasoning:
For z = 0, the output is 0.
For z = 1, the output is 1.
For z = -1, the output is -0.01 (0.01 * -1).
For z = -2 with alpha = 0.1, the output is -0.2 (0.1 * -2).

"""
"""
Understanding the Leaky ReLU Activation Function
The Leaky ReLU (Leaky Rectified Linear Unit) activation function is a variant of the ReLU function used in neural networks. It addresses the "dying ReLU" problem by allowing a small, non-zero gradient when the input is negative. This small slope for negative inputs helps keep the function active and prevents neurons from becoming inactive.

Mathematical Definition
The Leaky ReLU function is mathematically defined as:

f
(
z
)
=
{
z
if 
z
>
0
α
z
if 
z
≤
0
f(z)={ 
z
αz
​
  
if z>0
if z≤0
​
 
where 
z
z is the input to the function and 
α
α is a small positive constant, typically 
α
=
0.01
α=0.01.

In this definition, the function returns 
z
z for positive values, and for negative values, it returns 
α
z
αz, allowing a small gradient to pass through.

Characteristics
Output Range: The output is in the range 
(
−
∞
,
∞
)
(−∞,∞). Positive values are retained, while negative values are scaled by the factor 
α
α, allowing them to be slightly negative.
Shape: The function has a similar "L" shaped curve as ReLU, but with a small negative slope on the left side for negative 
z
z, creating a small gradient for negative inputs.
Gradient: The gradient is 1 for positive values of 
z
z and 
α
α for non-positive values. This allows the function to remain active even for negative inputs, unlike ReLU, where the gradient is zero for negative inputs.
This function is particularly useful in deep learning models as it mitigates the issue of "dead neurons" in ReLU by ensuring that neurons can still propagate a gradient even when the input is negative, helping to improve learning dynamics in the network.

"""
def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	# Your code here
	if z > 0 : return z 
    else : return alpha*z
