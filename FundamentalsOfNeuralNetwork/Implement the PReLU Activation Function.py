"""
Implement the PReLU (Parametric ReLU) activation function, a variant of the ReLU activation function that introduces a learnable parameter for negative inputs. Your task is to compute the PReLU activation value for a given input.

Example:
Input:
prelu(-2.0, alpha=0.25)
Output:
-0.5
Reasoning:
For x = -2.0 and alpha = 0.25, the PReLU activation is calculated as 
P
R
e
L
U
(
x
)
=
α
x
=
0.25
×
−
2.0
=
−
0.5
PReLU(x)=αx=0.25×−2.0=−0.5.

"""

def prelu(x: float, alpha: float = 0.25) -> float:
	"""
	Implements the PReLU (Parametric ReLU) activation function.

	Args:
		x: Input value
		alpha: Slope parameter for negative values (default: 0.25)

	Returns:
		float: PReLU activation value
	"""
	# Your code here
	if x > 0 :  return x 
    else : return alpha*x
