"""
Implement the Nesterov Accelerated Gradient (NAG) optimizer update step function. Your function should take the current parameter value, gradient function, and velocity as inputs, and return the updated parameter value and new velocity. The function should use the "look-ahead" approach where momentum is applied before computing the gradient, and should handle both scalar and array inputs.

Example:
Input:
parameter = 1.0, grad_fn = lambda x: x, velocity = 0.1
Output:
(0.9009, 0.0991)
Reasoning:
The Nesterov Accelerated Gradient optimizer computes updated values for the parameter and velocity using a look-ahead approach. With input values parameter=1.0, grad_fn=lambda x: x, and velocity=0.1, the updated parameter becomes 0.9009 and the updated velocity becomes 0.0991.

Learn About topic
Implementing Nesterov Accelerated Gradient (NAG) Optimizer
Introduction
Nesterov Accelerated Gradient (NAG) is an improvement over classical momentum optimization. While momentum helps accelerate gradient descent in the relevant direction, NAG takes this a step further by looking ahead in the direction of the momentum before computing the gradient. This "look-ahead" property helps NAG make more informed updates and often leads to better convergence.

Learning Objectives
Understand how Nesterov Accelerated Gradient optimization works
Learn to implement NAG-based gradient updates
Understand the advantages of NAG over classical momentum
Gain practical experience with advanced gradient-based optimization
Theory
Nesterov Accelerated Gradient uses a "look-ahead" approach where it first makes a momentum-based step and then computes the gradient at that position. The key equations are:

θ
l
o
o
k
a
h
e
a
d
,
t
−
1
=
θ
t
−
1
−
γ
v
t
−
1
θ 
lookahead,t−1
​
 =θ 
t−1
​
 −γv 
t−1
​
  (Look-ahead position)

v
t
=
γ
v
t
−
1
+
η
∇
θ
J
(
θ
l
o
o
k
a
h
e
a
d
,
t
−
1
)
v 
t
​
 =γv 
t−1
​
 +η∇ 
θ
​
 J(θ 
lookahead,t−1
​
 ) (Velocity update)

θ
t
=
θ
t
−
1
−
v
t
θ 
t
​
 =θ 
t−1
​
 −v 
t
​
  (Parameter update)

Where:

v
t
v 
t
​
  is the velocity at time t
γ
γ is the momentum coefficient (typically 0.9)
η
η is the learning rate
∇
θ
J
(
θ
)
∇ 
θ
​
 J(θ) is the gradient of the loss function
The key difference from classical momentum is that the gradient is evaluated at 
θ
l
o
o
k
a
h
e
a
d
,
t
−
1
θ 
lookahead,t−1
​
  instead of 
θ
t
−
1
θ 
t−1
​
 

Read more at:

Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate O(1/kÂ²). Doklady Akademii Nauk SSSR, 269(3), 543-547.
Ruder, S. (2017). An overview of gradient descent optimization algorithms. arXiv:1609.04747
Problem Statement
Implement the Nesterov Accelerated Gradient optimizer update step function. Your function should take the current parameter value, gradient function, and velocity as inputs, and return the updated parameter value and new velocity.

Input Format
The function should accept:

parameter: Current parameter value
gradient function: A function that accepts parameters and returns gradient computed at that point
velocity: Current velocity
learning_rate: Learning rate (default=0.01)
momentum: Momentum coefficient (default=0.9)
Output Format
Return tuple: (updated_parameter, updated_velocity)

Example
# Example usage:
def grad_func(parameter):
    # Returns gradient
    pass

parameter = 1.0
velocity = 0.1

new_param, new_velocity = nag_optimizer(parameter, grad_func, velocity)
Tips
Initialize velocity as zero
Use numpy for numerical operations
Test with both scalar and array inputs
Remember that the gradient should be computed at the look-ahead position

"""
import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the Nesterov Accelerated Gradient optimizer.
    Uses a "look-ahead" approach to improve convergence by applying momentum before computing the gradient.
    Args:
        parameter: Current parameter value
        grad_fn: Function that computes the gradient at a given position
        velocity: Current velocity (momentum term)
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
  
    param = np.array(parameter, dtype=np.float64)
    vel = np.array(velocity, dtype=np.float64)

    
    lookahead = param - momentum * vel
    gradient = np.array(grad_fn(lookahead), dtype=np.float64)
    new_velocity = momentum * vel + learning_rate * gradient
    updated_param = param - new_velocity
    if np.isscalar(parameter):
        updated_param = updated_param.item()
        new_velocity = new_velocity.item()

    return updated_param, new_velocity
