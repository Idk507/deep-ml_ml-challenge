"""
Implement a function that creates a simple residual block using NumPy. The block should take a 1D input array, process it through two weight layers (using matrix multiplication), apply ReLU activations, and add the original input via a shortcut connection before a final ReLU activation.

Example:
Input:
x = np.array([1.0, 2.0]), w1 = np.array([[1.0, 0.0], [0.0, 1.0]]), w2 = np.array([[0.5, 0.0], [0.0, 0.5]])
Output:
[1.5, 3.0]
Reasoning:
The input x is [1.0, 2.0]. First, compute w1 @ x = [1.0, 2.0], apply ReLU to get [1.0, 2.0]. Then, compute w2 @ [1.0, 2.0] = [0.5, 1.0]. Add the shortcut x to get [0.5 + 1.0, 1.0 + 2.0] = [1.5, 3.0]. Final ReLU gives [1.5, 3.0].

Learn About topic
Understanding Residual Blocks in ResNet
Residual blocks are the cornerstone of ResNet (Residual Network), a deep learning architecture designed to train very deep neural networks by addressing issues like vanishing gradients. The key idea is to allow the network to learn residuals differences between the input and the desired output rather than the full transformation.

Core Concept: Residual Learning
In a traditional neural network layer, the output is a direct transformation of the input, such as 
H
(
x
)
H(x), where 
x
x is the input. In a residual block, instead of learning 
H
(
x
)
H(x) directly, the network learns the residual 
F
(
x
)
=
H
(
x
)
−
x
F(x)=H(x)−x. The output of the block is then:

y
=
F
(
x
)
+
x
y=F(x)+x
Here, 
F
(
x
)
F(x) represents the transformation applied by the layers within the block (e.g., weight layers and activations), and 
x
x is the input, added back via a shortcut connection. This structure allows the network to learn an identity function (
F
(
x
)
=
0
F(x)=0, so 
y
=
x
y=x) if needed, which helps in training deeper networks.

Mathematical Structure
A typical residual block involves two weight layers with an activation function between them. The activation function used in ResNet is ReLU, defined as:

ReLU
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
ReLU(z)=max(0,z)
The block takes an input 
x
x, applies a transformation 
F
(
x
)
F(x) through the weight layers and activations, and then adds the input 
x
x back. Mathematically, if the weight layers are represented by matrices 
W
1
W 
1
​
  and 
W
2
W 
2
​
 , the transformation 
F
(
x
)
F(x) might look like a composition of operations involving 
W
1
⋅
x
W 
1
​
 ⋅x, a ReLU activation, and 
W
2
W 
2
​
  applied to the result. The final output 
y
y is the sum of 
F
(
x
)
F(x) and 
x
x, often followed by another ReLU activation to ensure non-negativity.

Why Shortcut Connections?
Ease of Learning: If the optimal transformation is close to an identity function, the block can learn 
F
(
x
)
≈
0
F(x)≈0, making 
y
≈
x
y≈x.
Gradient Flow: The shortcut connection allows gradients to flow directly through the addition operation during backpropagation, helping to train deeper networks without vanishing gradients.
Conceptual Example
Suppose the input 
x
x is a vector of length 2, and the weight layers are matrices 
W
1
W 
1
​
  and 
W
2
W 
2
​
 . The block computes 
F
(
x
)
F(x) by applying 
W
1
W 
1
​
 , a ReLU activation, and 
W
2
W 
2
​
 , then adds 
x
x to the result. The shortcut connection ensures that even if 
F
(
x
)
F(x) is small, the output 
y
y retains information from 
x
x, making it easier for the network to learn.

This structure is what enables ResNet to scale to hundreds of layers while maintaining performance, as shown in the diagram of the residual block.

"""
import numpy as np

def relu(x):
    return np.maximum(0,x)

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
    h1 = relu(w1 @ x)
    h2 = w2 @ h1 
    out = x + h2
    return relu(out)
	
