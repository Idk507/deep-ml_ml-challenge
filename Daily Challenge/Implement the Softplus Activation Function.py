"""
Implement the Softplus activation function, a smooth approximation of the ReLU function. Your task is to compute the Softplus value for a given input, handling edge cases to prevent numerical overflow or underflow.

Example:
Input:
softplus(2)
Output:
2.1269
Reasoning:
For x = 2, the Softplus activation is calculated as 
log⁡(1+ex)
Understanding the Softplus Activation Function
The Softplus activation function is a smooth approximation of the ReLU function. It's used in neural networks where a smoother transition around zero is desired. Unlike ReLU which has a sharp transition at x=0, Softplus provides a more gradual change.

Mathematical Definition
The Softplus function is mathematically defined as:

S
o
f
t
p
l
u
s
(
x
)
=
log
⁡
(
1
+
e
x
)
Softplus(x)=log(1+e 
x
 )
Where:

x
x is the input to the function
e
e is Euler's number (approximately 2.71828)
log
⁡
log is the natural logarithm
Characteristics
Output Range:

The output is always positive: 

(0,∞)
Unlike ReLU, Softplus never outputs exactly zero
Smoothness:

Softplus is continuously differentiable
The transition around x=0 is smooth, unlike ReLU's sharp "elbow"
Relationship to ReLU:

Softplus can be seen as a smooth approximation of ReLU
As x becomes very negative, Softplus approaches 0
As x becomes very positive, Softplus approaches x
Derivative:

The derivative of Softplus is the logistic sigmoid function:

​ d(Softplus(x))/d(x)= 1/1+e −x
​
 
Use Cases
When smooth gradients are important for optimization
In neural networks where a continuous approximation of ReLU is needed
Situations where strictly positive outputs are required with smooth transitions

"""

import numpy as np
def softplus(x: float) -> float:
	"""
	Compute the softplus activation function.

	Args:
		x: Input value

	Returns:
		The softplus value: log(1 + e^x)
	"""
	# Your code here
	return round(np.log(1+ np.exp(x)),4)
