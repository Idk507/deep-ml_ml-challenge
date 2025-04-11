"""
Write a Python function that computes the output of the sigmoid activation function given an input value z. The function should return the output rounded to four decimal places.

Example:
Input:
z = 0
Output:
0.5
Reasoning:
The sigmoid function is defined as σ(z) = 1 / (1 + exp(-z)). For z = 0, exp(-0) = 1, hence the output is 1 / (1 + 1) = 0.5.

Understanding the Sigmoid Activation Function
The sigmoid activation function is crucial in neural networks, especially for binary classification tasks. It maps any real-valued number into the interval ( (0, 1) ), making it useful for modeling probability as an output.

Mathematical Definition
The sigmoid function is mathematically defined as:

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
 
where ( z ) is the input to the function.

Characteristics
Output Range: The output is always between 0 and 1.
Shape: The function has an "S" shaped curve.
Gradient: The gradient is highest near ( z = 0 ) and decreases as ( z ) moves away from 0 in either direction.
The sigmoid function is particularly useful for turning logits (raw prediction values) into probabilities in binary classification models.

"""

import math

def sigmoid(z: float) -> float:
	exp = math.exp(-z)
	result = 1/(1+exp)
	return round(result,4)
