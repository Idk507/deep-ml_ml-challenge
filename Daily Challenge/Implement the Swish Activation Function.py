"""
Implement the Swish activation function, a self-gated activation function that has shown superior performance in deep neural networks compared to ReLU. Your task is to compute the Swish value for a given input.

Example:
Input:
swish(1)
Output:
0.7311
Reasoning:
For x = 1, the Swish activation is calculated as 
S
w
i
s
h
(
x
)
=
x
×
σ
(
x
)
Swish(x)=x×σ(x), where 
σ
(
x
)
=
1
1
+
e
−
x
σ(x)= 
1+e 
−x
 
1
​
 . Substituting the value, 
S
w
i
s
h
(
1
)
=
1
×
1
1
+
e
−
1
=
0.7311
Swish(1)=1× 
1+e 
−1
 
1
​
 =0.7311.
Understanding the Swish Activation Function
The Swish activation function is a modern self-gated activation function introduced by researchers at Google Brain. It has been shown to perform better than ReLU in many deep networks, particularly in deeper architectures.

Mathematical Definition
The Swish function is defined as:

S
w
i
s
h
(
x
)
=
x
×
σ
(
x
)
Swish(x)=x×σ(x)

where 
σ
(
x
)
σ(x) is the sigmoid function defined as:

σ
(
x
)
=
1
1
+
e
−
x
σ(x)= 
1+e 
−x
 
1
​
 

Characteristics
Output Range: Unlike ReLU which has a range of 
[
0
,
∞
)
[0,∞), Swish has a range of 
(
−
∞
,
∞
)
(−∞,∞)
Smoothness: Swish is smooth and non-monotonic, making it differentiable everywhere
Shape: The function has a slight dip below 0 for negative values, then curves up smoothly for positive values
Properties:
For large positive x: Swish(x) ~ x (similar to linear function)
For large negative x: Swish(x) ~ 0 (similar to ReLU)
Has a minimal value around x ~ -1.28
Advantages
Smooth function with no hard zero threshold like ReLU
Self-gated nature allows for more complex relationships
Often provides better performance in deep neural networks
Reduces the vanishing gradient problem compared to sigmoid
 """
def swish(x: float) -> float:
	"""
	Implements the Swish activation function.

	Args:
		x: Input value

	Returns:
		The Swish activation value
	"""
	# Your code here
    import numpy as np 
    s_x = 1/(1+np.exp(-x))
    ans = x * s_x 
    return ans
