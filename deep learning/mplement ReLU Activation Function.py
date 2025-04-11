"""
Write a Python function relu that implements the Rectified Linear Unit (ReLU) activation function. The function should take a single float as input and return the value after applying the ReLU function. The ReLU function returns the input if it's greater than 0, otherwise, it returns 0.

Example:
Input:
print(relu(0)) 
print(relu(1)) 
print(relu(-1))
Output:
0
1
0
Reasoning:
The ReLU function is applied to the input values 0, 1, and -1. The output is 0 for negative values and the input value for non-negative values.


Understanding the ReLU Activation Function
The ReLU (Rectified Linear Unit) activation function is widely used in neural networks, particularly in hidden layers of deep learning models. It maps any real-valued number to the non-negative range 
[
0
,
∞
)
[0,∞), which helps introduce non-linearity into the model while maintaining computational efficiency.

Mathematical Definition
The ReLU function is mathematically defined as:

f
(
z
)
=
max
⁡
(
0
,
z
)
f(z)=max(0,z)
where 
z
z is the input to the function.

Characteristics
Output Range: The output is always in the range 
[
0
,
∞
)
[0,∞). Values below 0 are mapped to 0, while positive values are retained.
Shape: The function has an "L" shaped curve with a horizontal axis at 
y
=
0
y=0 and a linear increase for positive 
z
z.
Gradient: The gradient is 1 for positive values of 
z
z and 0 for non-positive values. This means the function is linear for positive inputs and flat (zero gradient) for negative inputs.
This function is particularly useful in deep learning models as it introduces non-linearity while being computationally efficient, helping to capture complex patterns in the data.
"""

def relu(z: float) -> float:
	# Your code here
	return max(0,z)
