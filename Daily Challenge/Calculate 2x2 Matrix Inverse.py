"""
Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.

Example:
Input:
matrix = [[4, 7], [2, 6]]
Output:
[[0.6, -0.7], [-0.2, 0.4]]
Reasoning:
The inverse of a 2x2 matrix [a, b], [c, d] is given by (1/(ad-bc)) * [d, -b], [-c, a], provided ad-bc is not zero.

Calculating the Inverse of a 2x2 Matrix
The inverse of a matrix ( A ) is another matrix, often denoted ( A^{-1} ), such that:

A
A
−
1
=
A
−
1
A
=
I
AA 
−1
 =A 
−1
 A=I
where ( I ) is the identity matrix. For a 2x2 matrix:

A
=
(
a
b
c
d
)
A=( 
a
c
​
  
b
d
​
 )
The inverse is given by:

A
−
1
=
1
det
⁡
(
A
)
(
d
−
b
−
c
a
)
A 
−1
 = 
det(A)
1
​
 ( 
d
−c
​
  
−b
a
​
 )
provided that the determinant ( \det(A) = ad - bc ) is non-zero. If ( \det(A) = 0 ), the matrix does not have an inverse.

Importance
Calculating the inverse of a matrix is essential in various applications, such as solving systems of linear equations, where the inverse is used to find solutions efficiently.
"""

import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
	return np.linalg.inv(matrix)
