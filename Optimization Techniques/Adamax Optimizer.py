"""
Implement the Adamax optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages. The function should also handle scalar and array inputs and include bias correction for the moving averages.

Example:
Input:
parameter = 1.0, grad = 0.1, m = 0.0, u = 0.0, t = 1
Output:
(0.998, 0.01, 0.1)
Reasoning:
The Adamax optimizer computes updated values for the parameter, first moment (m), and infinity norm (u) using bias-corrected estimates of gradients. With input values parameter=1.0, grad=0.1, m=0.0, u=0.0, and t=1, the updated parameter becomes 0.998, the updated m becomes 0.01, and the updated u becomes 0.1.

"""
import numpy as np

def adamax_optimizer(parameter, grad, m, u, t, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adamax optimizer.
    Adamax is a variant of Adam based on the infinity norm.
    It uses the maximum of past squared gradients instead of the exponential moving average.
    Args:
        parameter: Current parameter value
        grad: Current gradient
        m: First moment estimate
        u: Infinity norm estimate
        t: Current timestep
        learning_rate: Learning rate (default=0.002)
        beta1: First moment decay rate (default=0.9)
        beta2: Infinity norm decay rate (default=0.999)
        epsilon: Small constant for numerical stability (default=1e-8)
    Returns:
        tuple: (updated_parameter, updated_m, updated_u)
    """
    assert learning_rate > 0, "Learning rate must be positive"
    assert 0 <= beta1 < 1, "Beta1 must be between 0 and 1"
    assert 0 <= beta2 < 1, "Beta2 must be between 0 and 1"
    assert epsilon > 0, "Epsilon must be positive"
    assert all(u >= 0) if isinstance(u, np.ndarray) else u >= 0, "u must be non-negative"

    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update infinity norm estimate
    u = np.maximum(beta2 * u, np.abs(grad))

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)

    # Update parameters
    update = learning_rate * m_hat / (u + epsilon)
    parameter = parameter - update

    return np.round(parameter, 5), np.round(m, 5), np.round(u, 5)
