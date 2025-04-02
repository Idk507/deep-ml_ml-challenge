"""
Implement the Hard Sigmoid activation function, a computationally efficient approximation of the standard sigmoid function. Your function should take a single input value and return the corresponding output based on the Hard Sigmoid definition.

Example:
Input:
hard_sigmoid(0.0)
Output:
0.5
Reasoning:
The input 0.0 falls in the linear region of the Hard Sigmoid function. Using the formula 
H
a
r
d
S
i
g
m
o
i
d
(
x
)
=
0.2
x
+
0.5
HardSigmoid(x)=0.2x+0.5, the output is 
0.2
×
0.0
+
0.5
=
0.5
0.2×0.0+0.5=0.5.

Learn About topic
Understanding the Hard Sigmoid Activation Function
The Hard Sigmoid is a piecewise linear approximation of the sigmoid activation function. It's computationally more efficient than the standard sigmoid function while maintaining similar characteristics. This function is particularly useful in neural networks where computational efficiency is crucial.

Mathematical Definition
The Hard Sigmoid function is mathematically defined as:

H
a
r
d
S
i
g
m
o
i
d
(
x
)
=
{
0
if 
x
≤
−
2.5
0.2
x
+
0.5
if 
−
2.5
<
x
<
2.5
1
if 
x
≥
2.5
HardSigmoid(x)= 
⎩
⎨
⎧
​
  
0
0.2x+0.5
1
​
  
if x≤−2.5
if −2.5<x<2.5
if x≥2.5
​
 
Where 
x
x is the input to the function.

Characteristics
Output Range: The output is always bounded in the range 
[
0
,
1
]
[0,1]
Shape: The function consists of three parts:
A constant value of 0 for inputs <= -2.5
A linear segment with slope 0.2 for inputs between -2.5 and 2.5
A constant value of 1 for inputs >= 2.5
Gradient: The gradient is 0.2 in the linear region and 0 in the saturated regions
Advantages in Neural Networks
This function is particularly useful in neural networks as it provides:

Computational efficiency compared to standard sigmoid
Bounded output range similar to sigmoid
Simple gradient computation

"""
def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function.

    Args:
        x (float): Input value

    Returns:
        float: The Hard Sigmoid of the input
    """
    if x <= -2.5:
        return 0.0
    elif x >= 2.5:
        return 1.0
    else:
        return 0.2 * x + 0.5
