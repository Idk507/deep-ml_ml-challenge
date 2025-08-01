"""
Implement the momentum optimizer update step function. Your function should take the current parameter value, gradient, and velocity as inputs, and return the updated parameter value and new velocity. The function should also handle scalar and array inputs.

Example:
Input:
parameter = 1.0, grad = 0.1, velocity = 0.1
Output:
(0.909, 0.091)
Reasoning:
The momentum optimizer computes updated values for the parameter and the velocity. With input values parameter=1.0, grad=0.1, and velocity=0.1, the updated parameter becomes 0.909 and the updated velocity becomes 0.091.

Implementing Momentum Optimizer
Introduction
Momentum is a popular optimization technique that helps accelerate gradient descent in the relevant direction and dampens oscillations. It works by adding a fraction of the previous update vector to the current gradient.

Learning Objectives
Understand how momentum optimization works
Learn to implement momentum-based gradient updates
Understand the effect of momentum on optimization
Theory
Momentum optimization uses a moving average of gradients to determine the direction of the update. The key equations are:

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
 J(θ) (Velocity update)

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
Read more at:

Ruder, S. (2017). An overview of gradient descent optimization algorithms. arXiv:1609.04747
Problem Statement
Implement the momentum optimizer update step function. Your function should take the current parameter value, gradient, and velocity as inputs, and return the updated parameter value and new velocity.

Input Format
The function should accept:

parameter: Current parameter value
grad: Current gradient
velocity: Current velocity
learning_rate: Learning rate (default=0.01)
momentum: Momentum coefficient (default=0.9)
Output Format
Return tuple: (updated_parameter, updated_velocity)

Example
# Example usage:
parameter = 1.0
grad = 0.1
velocity = 0.1

new_param, new_velocity = momentum_optimizer(parameter, grad, velocity)
Tips
Initialize velocity as zero
Use numpy for numerical operations
Test with both scalar and array inputs

"""
import numpy as np

def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the momentum optimizer.
    Uses momentum to accelerate learning in relevant directions and dampen oscillations.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        velocity: Current velocity/momentum term
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    assert learning_rate > 0, "Learning rate must be positive"
    assert 0 <= momentum < 1, "Momentum must be between 0 and 1"

    # Update velocity
    velocity = momentum * velocity + learning_rate * grad

    # Update parameters
    parameter = parameter - velocity

    return np.round(parameter, 5), np.round(velocity, 5)
