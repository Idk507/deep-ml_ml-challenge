"""
Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b. The function should iterate n times, rounding each intermediate solution to four decimal places, and return the approximate solution x.

Example:
Input:
A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2
Output:
[0.146, 0.2032, -0.5175]
Reasoning:
The Jacobi method iteratively solves each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)), where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.

Solving Linear Equations Using the Jacobi Method
The Jacobi method is an iterative algorithm used for solving a system of linear equations ( Ax = b ). This method is particularly useful for large systems where direct methods, such as Gaussian elimination, are computationally expensive.

Algorithm Overview
For a system of equations represented by ( Ax = b ), where ( A ) is a matrix and ( x ) and ( b ) are vectors, the Jacobi method involves the following steps:
Initialization: Start with an initial guess for ( x ).

Iteration: For each equation ( i ), update ( x[i] ) using:

x
[
i
]
=
1
a
i
i
(
b
[
i
]
−
∑
j
≠
i
a
i
j
x
[
j
]
)
x[i]= 
a 
ii
​
 
1
​
  
​
 b[i]− 
j

=i
∑
​
 a 
ij
​
 x[j] 
​
 
where ( a_{ii} ) are the diagonal elements of ( A ), and ( a_{ij} ) are the off-diagonal elements.

Convergence: Repeat the iteration until the changes in ( x ) are below a certain tolerance or until a maximum number of iterations is reached.

This method assumes that all diagonal elements of ( A ) are non-zero and that the matrix is diagonally dominant or properly conditioned for convergence.

Practical Considerations
The method may not converge for all matrices.
Choosing a good initial guess can improve convergence.
Diagonal dominance of ( A ) ensures the convergence of the Jacobi method.

"""

import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    d_a = np.diag(A)
    nda = A - np.diag(d_a)
    x = np.zeros(len(b))
    x_hold = np.zeros(len(b))
    for _ in range(n):
        for i in range(len(A)):
            x_hold[i] = (1/d_a[i]) * (b[i] - sum(nda[i]*x))
        x = x_hold.copy()
    return np.round(x,4).tolist()

