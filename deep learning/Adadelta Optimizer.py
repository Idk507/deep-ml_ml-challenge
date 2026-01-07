   """
Adadelta Optimizer
Medium
Deep Learning  

Implement the Adadelta optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages. The function should handle both scalar and array inputs, and include proper input validation.

Example:
Input:
parameter = 1.0, grad = 0.1, u = 1.0, v = 1.0, rho = 0.95, epsilon = 1e-6   
Output:   
(0.89743, 0.9505, 0.95053)
Reasoning:
Learn About topic
Implementing Adadelta Optimizer
Introduction
Adadelta is an extension of Adagrad that addresses two key issues: the aggressive, monotonically decreasing learning rate and the need for manual learning rate tuning. While Adagrad accumulates all past squared gradients, Adadelta restricts the influence of past gradients to a window of size w. Instead of explicitly storing w past gradients, it efficiently approximates this window using an exponential moving average with decay rate Ï, making it more robust to parameter updates. Additionally, it automatically handles the units of the updates, eliminating the need for a manually set learning rate.

Learning Objectives
Understand how Adadelta optimizer works
Learn to implement adaptive learning rates with moving averages
Theory
Adadelta uses two main ideas:

Exponential moving average of squared gradients to approximate a window of size w
Automatic unit correction through the ratio of parameter updates
The key equations are:

v
t
=
ρ
v
t
−
1
+
(
1
−
ρ
)
g
t
2
v 
t
​
 =ρv 
t−1
​
 +(1−ρ)g 
t
2
​
  (Exponential moving average of squared gradients)

The above approximates a window size of 
w
≈
1
1
−
ρ
w≈ 
1−ρ
1
​
 

Δ
θ
t
=
−
u
t
−
1
+
ϵ
v
t
+
ϵ
⋅
g
t
Δθ 
t
​
 =− 
v 
t
​
 +ϵ
​
 
u 
t−1
​
 +ϵ
​
 
​
 ⋅g 
t
​
  (Parameter update with unit correction)

u
t
=
ρ
u
t
−
1
+
(
1
−
ρ
)
Δ
θ
t
2
u 
t
​
 =ρu 
t−1
​
 +(1−ρ)Δθ 
t
2
​
  (Exponential moving average of squared parameter updates)

Where:

v
t
v 
t
​
  is the exponential moving average of squared gradients (decay rate Ï)
u
t
u 
t
​
  is the exponential moving average of squared parameter updates (decay rate Ï)
ρ
ρ is the decay rate (typically 0.9) that controls the effective window size w â 1/(1-Ï)
ϵ
ϵ is a small constant for numerical stability
g
t
g 
t
​
  is the gradient at time step t
The ratio 
u
t
−
1
+
ϵ
v
t
+
ϵ
v 
t
​
 +ϵ
​
 
u 
t−1
​
 +ϵ
​
 
​
  serves as an adaptive learning rate that automatically handles the units of the updates, making the algorithm more robust to different parameter scales. Unlike Adagrad, Adadelta does not require a manually set learning rate, making it especially useful when tuning hyperparameters is difficult. This automatic learning rate adaptation is achieved through the ratio of the root mean squared (RMS) of parameter updates to the RMS of gradients.

Read more at:

Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. arXiv:1212.5701
Ruder, S. (2017). An overview of gradient descent optimization algorithms. arXiv:1609.04747
Problem Statement
Implement the Adadelta optimizer update step function. Your function should take the current parameter value, gradient, and accumulated statistics as inputs, and return the updated parameter value and new accumulated statistics.

Input Format
The function should accept:

parameter: Current parameter value
grad: Current gradient
v: Exponentially decaying average of squared gradients
u: Exponentially decaying average of squared parameter updates
rho: Decay rate (default=0.9)
epsilon: Small constant for numerical stability (default=1e-8)
Output Format
Return tuple: (updated_parameter, updated_v, updated_u)

Example
# Example usage:
parameter = 1.0
grad = 0.1
v = 1.0
u = 1.0

new_param, new_v, new_u = adadelta_optimizer(parameter, grad, v, u)
Tips
Initialize v and u as zeros
Use numpy for numerical operations
Test with both scalar and array inputs
The learning rate is automatically determined by the algorithm


"""

import numpy as np

def adadelta_optimizer(parameter, grad, v, u, rho=0.9, epsilon=1e-8):
    """
    Adadelta optimizer update step.
    
    Parameters:
    -----------
    parameter : float or np.ndarray
        Current parameter value(s)
    grad : float or np.ndarray
        Current gradient(s)
    v : float or np.ndarray
        Exponentially decaying average of squared gradients
    u : float or np.ndarray
        Exponentially decaying average of squared parameter updates
    rho : float, default=0.9
        Decay rate for moving averages (controls window size ≈ 1/(1-rho))
    epsilon : float, default=1e-8
        Small constant for numerical stability
    
    Returns:
    --------
    tuple: (updated_parameter, updated_v, updated_u)
        - updated_parameter: New parameter value(s)
        - updated_v: New exponential moving average of squared gradients
        - updated_u: New exponential moving average of squared parameter updates
    
    Notes:
    ------
    The Adadelta algorithm updates parameters using:
    1. v_t = rho * v_{t-1} + (1 - rho) * grad^2
    2. delta_theta = -sqrt(u_{t-1} + epsilon) / sqrt(v_t + epsilon) * grad
    3. u_t = rho * u_{t-1} + (1 - rho) * delta_theta^2
    4. theta_t = theta_{t-1} + delta_theta
    """
    # Input validation
    if not isinstance(rho, (int, float)) or not (0 <= rho < 1):
        raise ValueError("rho must be a number between 0 and 1")
    
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("epsilon must be a positive number")
    
    # Convert inputs to numpy arrays for consistent handling
    parameter = np.asarray(parameter, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    
    # Check shape compatibility
    if parameter.shape != grad.shape:
        raise ValueError(f"parameter and grad must have the same shape. "
                        f"Got parameter: {parameter.shape}, grad: {grad.shape}")
    
    if v.shape != parameter.shape:
        raise ValueError(f"v must have the same shape as parameter. "
                        f"Got v: {v.shape}, parameter: {parameter.shape}")
    
    if u.shape != parameter.shape:
        raise ValueError(f"u must have the same shape as parameter. "
                        f"Got u: {u.shape}, parameter: {parameter.shape}")
    
    # Step 1: Update exponential moving average of squared gradients
    # v_t = rho * v_{t-1} + (1 - rho) * grad^2
    updated_v = rho * v + (1 - rho) * grad**2
    
    # Step 2: Calculate parameter update with automatic learning rate
    # delta_theta = -sqrt(u_{t-1} + epsilon) / sqrt(v_t + epsilon) * grad
    delta_theta = -np.sqrt(u + epsilon) / np.sqrt(updated_v + epsilon) * grad
    
    # Step 3: Update exponential moving average of squared parameter updates
    # u_t = rho * u_{t-1} + (1 - rho) * delta_theta^2
    updated_u = rho * u + (1 - rho) * delta_theta**2
    
    # Step 4: Update parameter
    # theta_t = theta_{t-1} + delta_theta
    updated_parameter = parameter + delta_theta
    
    # Convert back to scalar if input was scalar
    if updated_parameter.shape == ():
        return (float(updated_parameter), float(updated_v), float(updated_u))
    
    return (updated_parameter, updated_v, updated_u)


