"""
Implement a single update step of the AdamW optimizer for a parameter vector 
w
w and its gradients 
g
g. AdamW is a variant of the Adam optimizer that decouples weight decay from the gradient update, leading to better generalization.

Your Task: Write a function adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay) that performs one update step for the parameter vector 
w
w with its gradient 
g
g using AdamW. The function should:

Update the first moment 
m
m and second moment 
v
v (moving averages of gradients and squared gradients)
Apply bias correction for 
m
m and 
v
v
Apply the AdamW update rule (with decoupled weight decay)
Return the updated parameter vector and the new values of 
m
m and 
v
v
Arguments:

w: NumPy array, current parameter vector
g: NumPy array, gradient vector (same shape as w)
m: NumPy array, first moment vector (same shape as w)
v: NumPy array, second moment vector (same shape as w)
t: Integer, current time step (starting from 1)
lr: Learning rate (float)
beta1: Decay rate for the first moment (float)
beta2: Decay rate for the second moment (float)
epsilon: Small constant for numerical stability (float)
weight_decay: Weight decay coefficient (float)
Example:
Input:
import numpy as np
w = np.array([1.0, 2.0])
g = np.array([0.1, -0.2])
m = np.zeros(2)
v = np.zeros(2)
t = 1
lr = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.1
w_new, m_new, v_new = adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay)
print(np.round(w_new, 4))
Output:
[0.989 2.008]
Reasoning:
After applying AdamW update, the weights are moved in the negative gradient direction and decayed by 1%. The result is [0.989, 2.001].

"""
import numpy as np

def adamw_update(w, g, m, v, t, lr, beta1, beta2, epsilon, weight_decay):
    """
    Perform one AdamW optimizer step.
    
    Args:
      w: parameter vector (np.ndarray)
      g: gradient vector (np.ndarray)
      m: first moment vector (np.ndarray)
      v: second moment vector (np.ndarray)
      t: integer, current time step
      lr: float, learning rate
      beta1: float, beta1 parameter
      beta2: float, beta2 parameter
      epsilon: float, small constant
      weight_decay: float, weight decay coefficient
      
    Returns:
      w_new, m_new, v_new
    """
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * g
    
    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (g ** 2)
    
    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1 ** t)
    
    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2 ** t)
    
    # Compute parameter update
    w = w - lr * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * w)
    
    return w, m, v
