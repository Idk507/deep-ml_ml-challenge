"""
Implement a Python function that applies the SwiGLU activation function to a NumPy array. Assume the input array has already been passed through a linear projection and has shape (batch_size, 2d). Round each output to four decimal places and return the result as a NumPy array of the shape (batch_size, d).

Example:
Input:
np.array([[1, -1, 1000, -1000]])
Output:
[[1000., 0.]]
Reasoning:
The input is of shape (1, 4), so it is split into x1 = [1, -1] and x2 = [1000, -1000]. The sigmoid of 1000 is approximately 1, and the sigmoid of -1000 is approximately 0 due to saturation. Thus, Swish(1000) ≈ 1000 x 1 = 1000 and Swish(-1000) ≈ -1000 x 0 = 0. Then, SwiGLU = x1 * Swish(x2) = [1 x 1000, -1 x 0] = [1000, 0].

"""
import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Your code here
    def silu(x):
        sigmoid = 1 / (1 + np.exp(-x))
        swish = x * sigmoid 
        return swish
    assert x.shape[-1] % 2 == 0, "Last dimension must be even to split into two halves"
    d = x.shape[-1] // 2
    x1,x2 = x[...,:d],x[...,d:]
    return x1 * silu(x2)
