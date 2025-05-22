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

def conjugate_gradient(A: np.array, b: np.array, n: int, x0: np.array=None, tol=1e-8) -> np.array:

    # calculate initial residual vector
    x = np.zeros_like(b)
    r = residual(A, b, x) # residual vector
    rPlus1 = r
    p = r # search direction vector

    for i in range(n):

        # line search step value - this minimizes the error along the current search direction
        alp = alpha(A, r, p)

        # new x and r based on current p (the search direction vector)
        x = x + alp * p
        rPlus1 = r - alp * (A@p)

        # calculate beta - this ensures that all vectors are A-orthogonal to each other
        bet = beta(r, rPlus1)

        # update x and r
        # using a othogonal search direction ensures we get all the information we need in more direction and then don't have to search in that direction again
        p = rPlus1 + bet * p

        # update residual vector
        r = rPlus1

        # break if less than tolerance
        if np.linalg.norm(residual(A,b,x)) < tol:
            break

    return x

def residual(A: np.array, b: np.array, x: np.array) -> np.array:
    # calculate linear system residuals
    return b - A @ x

def alpha(A: np.array, r: np.array, p: np.array) -> float:

    # calculate step size
    alpha_num = np.dot(r, r)
    alpha_den = np.dot(p @ A, p)

    return alpha_num/alpha_den

def beta(r: np.array, r_plus1: np.array) -> float:

    # calculate direction scaling
    beta_num = np.dot(r_plus1, r_plus1)
    beta_den = np.dot(r, r)

    return beta_num/beta_den
