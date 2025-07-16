"""
Implement the PReLU Activation Function

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

Understanding the PReLU (Parametric ReLU) Activation Function
The PReLU (Parametric Rectified Linear Unit) is an advanced variant of the ReLU activation function that introduces a learnable parameter for negative inputs. This makes it more flexible than standard ReLU and helps prevent the "dying ReLU" problem.

Mathematical Definition
The PReLU function is defined as:

P
R
e
L
U
(
x
)
=
{
x
if 
x
>
0
α
x
otherwise
PReLU(x)={ 
x
αx
​
  
if x>0
otherwise
​
 
Where:

x
x is the input value
α
α is a learnable parameter (typically initialized to a small value like 0.25)
Key Characteristics
Adaptive Slope: Unlike ReLU which has a zero slope for negative inputs, PReLU learns the optimal negative slope parameter (
α
α) during training.

Output Range:

For 
x
>
0
x>0: Output equals input (
y
=
x
y=x)
For 
x
≤
0
x≤0: Output is scaled by 
α
α (
y
=
α
x
y=αx)
Advantages:

Helps prevent the "dying ReLU" problem
More flexible than standard ReLU
Can improve model performance through learned parameter
Maintains the computational efficiency of ReLU
Special Cases:

When 
α
=
0
α=0, PReLU becomes ReLU
When 
α
=
1
α=1, PReLU becomes a linear function
When 
α
α is small (e.g., 0.01), PReLU behaves similarly to Leaky ReLU
PReLU is particularly useful in deep neural networks where the optimal negative slope might vary across different layers or channels.

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
	if x > 0 :
        return x
    else :
        return alpha * x
