""""
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

