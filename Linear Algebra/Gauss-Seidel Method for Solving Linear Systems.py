"""
Task: Implement the Gauss-Seidel Method
Your task is to implement the Gauss-Seidel method, an iterative technique for solving a system of linear equations (Ax = b).

The function should iteratively update the solution vector (x) by using the most recent values available during the iteration process.

Write a function gauss_seidel(A, b, n, x_ini=None) where:

A is a square matrix of coefficients,
b is the right-hand side vector,
n is the number of iterations,
x_ini is an optional initial guess for (x) (if not provided, assume a vector of zeros).
The function should return the approximated solution vector (x) after performing the specified number of iterations.

Example:
Input:
A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float)
b = np.array([4, 7, 3], dtype=float)

n = 100
print(gauss_seidel(A, b, n))
Output:
# [0.2, 1.4, 0.8]  (Approximate, values may vary depending on iterations)
Reasoning:
The Gauss-Seidel method iteratively updates the solution vector (x) until convergence. The output is an approximate solution to the linear system.

Understanding the Gauss-Seidel Method
The Gauss-Seidel method is a technique for solving linear systems of equations 
A
x
=
b
Ax=b. Unlike fixed-point Jacobi, Gauss-Seidel uses previously computed results as soon as they are available. This increases convergence, resulting in fewer iterations, but it is not as easily parallelizable as fixed-point Jacobi.

Mathematical Formulation
Initialization: Start with an initial guess for 
x
x.

Iteration: For each equation 
i
i, update 
x
[
i
]
x[i] using:

x
i
(
k
+
1
)
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
<
i
a
i
j
x
j
(
k
+
1
)
−
∑
j
>
i
a
i
j
x
j
(
k
)
)
x 
i
(k+1)
​
 = 
a 
ii
​
 
1
​
 (b[i]− 
j<i
∑
​
 a 
ij
​
 x 
j
(k+1)
​
 − 
j>i
∑
​
 a 
ij
​
 x 
j
(k)
​
 )
where 
a
i
i
a 
ii
​
  represents the diagonal elements of 
A
A, and 
a
i
j
a 
ij
​
  represents the off-diagonal elements.

Convergence: Repeat the iteration until the changes in 
x
x are below a set tolerance or until a maximum number of iterations is reached.
Matrix Form
The Gauss-Seidel method can also be expressed in matrix form using the diagonal matrix 
D
D, lower triangle 
L
L, and upper triangle 
U
U:

x
(
k
+
1
)
=
D
−
1
(
b
−
L
x
(
k
+
1
)
−
U
x
(
k
)
)
x 
(k+1)
 =D 
−1
 (b−Lx 
(k+1)
 −Ux 
(k)
 )
Example Calculation
Letâs solve the system of equations:

3
x
1
+
x
2
=
5
x
1
+
2
x
2
=
5
3x 
1
​
 +x 
2
​
 =5x 
1
​
 +2x 
2
​
 =5
Initialize 
x
1
(
0
)
=
0
x 
1
(0)
​
 =0 and 
x
2
(
0
)
=
0
x 
2
(0)
​
 =0.

First iteration:

For 
x
1
(
1
)
x 
1
(1)
​
 :

x
1
(
1
)
=
1
3
(
5
−
1
⋅
x
2
(
0
)
)
=
5
3
≈
1.6667
x 
1
(1)
​
 = 
3
1
​
 (5−1⋅x 
2
(0)
​
 )= 
3
5
​
 ≈1.6667
For 
x
2
(
1
)
x 
2
(1)
​
 :

x
2
(
1
)
=
1
2
(
5
−
1
⋅
x
1
(
1
)
)
=
1
2
(
5
−
1.6667
)
≈
1.6667
x 
2
(1)
​
 = 
2
1
​
 (5−1⋅x 
1
(1)
​
 )= 
2
1
​
 (5−1.6667)≈1.6667
After the first iteration, the values are 
x
1
(
1
)
=
1.6667
x 
1
(1)
​
 =1.6667 and 
x
2
(
1
)
=
1.6667
x 
2
(1)
​
 =1.6667.

Continue iterating until the results converge to a desired tolerance.

Applications
The Gauss-Seidel method and other iterative solvers are commonly used in data science, computational fluid dynamics, and 3D graphics.
"""
import numpy as np

def gauss_seidel_it(A, b, x):
    rows, cols = A.shape
    for i in range(rows):
        x_new = b[i]
        for j in range(cols):
            if i != j:
                x_new -= A[i, j] * x[j]
        x[i] = x_new / A[i, i]
    return x

def gauss_seidel(A, b, n, x_ini=None):
    x = x_ini or np.zeros_like(b)
    for _ in range(n):
        x = gauss_seidel_it(A, b, x)
    return x
