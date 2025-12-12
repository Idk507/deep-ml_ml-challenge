"""
Implement Newton's method for finding the minimum of a function. Given functions that compute the gradient and Hessian at any point, iteratively update the position using the Newton step until convergence. Newton's method uses second-order information (curvature) to converge faster than gradient descent, often finding the minimum of quadratic functions in a single step.

Example:
Input:
f(x,y) = (x-1)^2 + (y-2)^2, gradient_func, hessian_func, x0 = [0.0, 0.0]
Output:
[1.0, 2.0]
Reasoning:
At x0=[0,0]: grad=[-2,-4], Hessian=[[2,0],[0,2]]. Newton step: delta = -H^{-1}grad = -[[0.5,0],[0,0.5]][-2,-4] = [1,2]. New point: [0,0]+[1,2]=[1,2]. Since this is a quadratic function, Newton's method converges in exactly one step to the minimum at (1,2).


"""
import numpy as np
from typing import Callable

def newtons_method_optimization(
    gradient_func: Callable[[list[float]], list[float]],
    hessian_func: Callable[[list[float]], list[list[float]]],
    x0: list[float],
    tol: float = 1e-6,
    max_iter: int = 100
) -> list[float]:
    """
    Find the minimum of a function using Newton's method.
    
    Args:
        gradient_func: Function that returns gradient vector at a point
        hessian_func: Function that returns Hessian matrix at a point
        x0: Initial guess (list of coordinates)
        tol: Convergence tolerance for gradient norm
        max_iter: Maximum number of iterations
        
    Returns:
        The point that minimizes the function
    """
    x = np.array(x0, dtype=float)
    
    for _ in range(max_iter):
        grad = np.array(gradient_func(x.tolist()))
        hess = np.array(hessian_func(x.tolist()))
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
        
        try:
            # Solve Newton step: delta = -H^{-1} grad
            delta = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            raise ValueError("Hessian is singular and cannot be inverted.")
        
        x = x + delta
    
    return x.tolist()
