"""
Implement the Adam optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages. The function should also handle scalar and array inputs and include bias correction for the moving averages.

Example:
Input:
parameter = 1.0, grad = 0.1, m = 0.0, v = 0.0, t = 1
Output:
(0.999, 0.01, 0.0001)
Reasoning:
The Adam optimizer computes updated values for the parameter, first moment (m), and second moment (v) using bias-corrected estimates of gradients. With input values parameter=1.0, grad=0.1, m=0.0, v=0.0, and t=1, the updated parameter becomes 0.999.

Implementing Adam Optimizer
Introduction
Adam (Adaptive Moment Estimation) is a popular optimization algorithm used in training deep learning models. It combines the benefits of two other optimization algorithms: RMSprop and momentum optimization.

Learning Objectives
Understand how Adam optimizer works
Learn to implement adaptive learning rates
Understand bias correction in optimization algorithms
Gain practical experience with gradient-based optimization
Theory
Adam maintains moving averages of both gradients (first moment) and squared gradients (second moment) to adapt the learning rate for each parameter. The key equations are:

m
t
=
β
1
m
t
−
1
+
(
1
−
β
1
)
g
t
m 
t
​
 =β 
1
​
 m 
t−1
​
 +(1−β 
1
​
 )g 
t
​
  (First moment) 
v
t
=
β
2
v
t
−
1
+
(
1
−
β
2
)
g
t
2
v 
t
​
 =β 
2
​
 v 
t−1
​
 +(1−β 
2
​
 )g 
t
2
​
  (Second moment)

Bias correction: 
m
^
t
=
m
t
1
−
β
1
t
m
^
  
t
​
 = 
1−β 
1
t
​
 
m 
t
​
 
​
  
v
^
t
=
v
t
1
−
β
2
t
v
^
  
t
​
 = 
1−β 
2
t
​
 
v 
t
​
 
​
 

Parameter update: 
θ
t
=
θ
t
−
1
−
α
m
^
t
v
^
t
+
ϵ
θ 
t
​
 =θ 
t−1
​
 −α 
v
^
  
t
​
 
​
 +ϵ
m
^
  
t
​
 
​
 

Problem Statement
Implement the Adam optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages.

Input Format
The function should accept:

parameter: Current parameter value
grad: Current gradient
m: First moment estimate
v: Second moment estimate
t: Current timestep
learning_rate: Learning rate (default=0.001)
beta1: First moment decay rate (default=0.9)
beta2: Second moment decay rate (default=0.999)
epsilon: Small constant for numerical stability (default=1e-8)
Output Format
Return tuple: (updated_parameter, updated_m, updated_v)

Example
Example usage:
parameter = 1.0 grad = 0.1 m = 0.0 v = 0.0 t = 1

new_param, new_m, new_v = adam_optimizer(parameter, grad, m, v, t)

Tips
Initialize m and v as zeros
Keep track of timestep t for bias correction
Use numpy for numerical operations
Test with both scalar and array inputs

"""

import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adam optimizer.
    Adjusts the learning rate based on the moving averages of the gradient and squared gradient.
    :param parameter: Current parameter value
    :param grad: Current gradient
    :param m: First moment estimate
    :param v: Second moment estimate
    :param t: Current timestep
    :param learning_rate: Learning rate (default=0.001)
    :param beta1: First moment decay rate (default=0.9)
    :param beta2: Second moment decay rate (default=0.999)
    :param epsilon: Small constant for numerical stability (default=1e-8)
    :return: tuple: (updated_parameter, updated_m, updated_v)
    """
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (grad**2)

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)

    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2**t)

    # Update parameters
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    parameter = parameter - update

    return np.round(parameter,5), np.round(m,5), np.round(v,5)
