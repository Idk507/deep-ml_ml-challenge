"""
Implement the Conjugate Gradient Method for Solving Linear Systems

Task: Implement the Conjugate Gradient Method for Solving Linear Systems
Your task is to implement the Conjugate Gradient (CG) method, an efficient iterative algorithm for solving large, sparse, symmetric, positive-definite linear systems. Given a matrix A and a vector b, the algorithm will solve for x in the system ( Ax = b ).

Write a function conjugate_gradient(A, b, n, x0=None, tol=1e-8) that performs the Conjugate Gradient method as follows:

A: A symmetric, positive-definite matrix representing the linear system.
b: The vector on the right side of the equation.
n: Maximum number of iterations.
x0: Initial guess for the solution vector.
tol: Tolerance for stopping criteria.
The function should return the solution vector x.

Example:
Input:
A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])
n = 5

print(conjugate_gradient(A, b, n))
Output:
[0.09090909, 0.63636364]
Reasoning:
The Conjugate Gradient method is applied to the linear system Ax = b with the given matrix A and vector b. The algorithm iteratively refines the solution to converge to the exact solution.



""""
import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solves the system Ax = b using the Conjugate Gradient method.

    Parameters:
    A (ndarray): Symmetric positive-definite matrix.
    b (ndarray): Right-hand side vector.
    n (int): Maximum number of iterations.
    x0 (ndarray): Initial guess for the solution (optional).
    tol (float): Tolerance for the stopping condition.

    Returns:
    x (ndarray): The approximate solution to Ax = b.
    """
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0

    r = b - A @ x       # Initial residual
    p = r.copy()        # Initial direction
    rs_old = np.dot(r, r)

    for i in range(n):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)

        x = x + alpha * p          # Update estimate
        r = r - alpha * Ap         # Update residual

        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:  # Check convergence
            break

        beta = rs_new / rs_old     # Compute scaling factor
        p = r + beta * p           # Update direction
        rs_old = rs_new

    return x

