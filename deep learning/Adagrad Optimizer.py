"""
Implement the Adagrad optimizer update step function. Your function should take the current parameter value, gradient, and accumulated squared gradients as inputs, and return the updated parameter value and new accumulated squared gradients. The function should also handle scalar and array inputs, and include proper input validation.

Example:
Input:
parameter = 1.0, grad = 0.1, G = 1.0
Output:
(0.999, 1.01)
Reasoning:
The Adagrad optimizer computes updated values for the parameter and the accumulated squared gradients. With input values parameter=1.0, grad=0.1, and G=1.0, the updated parameter becomes 0.999 and the updated G becomes 1.01.



Introduction
Adagrad (Adaptive Gradient Algorithm) is an optimization algorithm that adapts the learning rate to each parameter, performing larger updates for infrequent parameters and smaller updates for frequent ones. This makes it particularly well-suited for dealing with sparse data.

Learning Objectives
Understand how Adagrad optimizer works
Learn to implement adaptive learning rates
Gain practical experience with gradient-based optimization
Theory
Adagrad adapts the learning rate for each parameter based on the historical gradients. The key equations are:

G
t
=
G
t
−
1
+
g
t
2
G 
t
​
 =G 
t−1
​
 +g 
t
2
​
  (Accumulated squared gradients)

θ
t
=
θ
t
−
1
−
α
G
t
+
ϵ
⋅
g
t
θ 
t
​
 =θ 
t−1
​
 − 
G 
t
​
 
​
 +ϵ
α
​
 ⋅g 
t
​
  (Parameter update)

Where:

G
t
G 
t
​
  is the sum of squared gradients up to time step t
α
α is the initial learning rate
ϵ
ϵ is a small constant for numerical stability
g
t
g 
t
​
  is the gradient at time step t
Read more at:

Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12, 2121â2159. PDF
Ruder, S. (2017). An overview of gradient descent optimization algorithms. arXiv:1609.04747
Problem Statement
Implement the Adagrad optimizer update step function. Your function should take the current parameter value, gradient, and accumulated squared gradients as inputs, and return the updated parameter value and new accumulated squared gradients.

Input Format
The function should accept:

parameter: Current parameter value
grad: Current gradient
G: Accumulated squared gradients
learning_rate: Learning rate (default=0.01)
epsilon: Small constant for numerical stability (default=1e-8)
Output Format
Return tuple: (updated_parameter, updated_G)

Example
# Example usage:
parameter = 1.0
grad = 0.1
G = 1.0

new_param, new_G = adagrad_optimizer(parameter, grad, G)
Tips
Initialize G as zeros
Use numpy for numerical operations
Test with both scalar and array inputs


"""

import numpy as np

def adagrad_optimizer(parameter, grad, G, learning_rate=0.01, epsilon=1e-8):
    """
    Performs one Adagrad update step.

    Parameters:
    - parameter: scalar or np.ndarray, current parameter value
    - grad: scalar or np.ndarray, current gradient
    - G: scalar or np.ndarray, accumulated squared gradients
    - learning_rate: float, learning rate (default=0.01)
    - epsilon: float, small constant for numerical stability (default=1e-8)

    Returns:
    - tuple: (updated_parameter, updated_G)
    """
    # Convert inputs to numpy arrays for consistency
    parameter = np.asarray(parameter, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)

    # Input validation
    if parameter.shape != grad.shape or parameter.shape != G.shape:
        raise ValueError("parameter, grad, and G must have the same shape")

    # Update accumulated squared gradients
    updated_G = G + grad**2

    # Update parameter
    adjusted_lr = learning_rate / (np.sqrt(updated_G) + epsilon)
    updated_parameter = parameter - adjusted_lr * grad

    return updated_parameter, updated_G
