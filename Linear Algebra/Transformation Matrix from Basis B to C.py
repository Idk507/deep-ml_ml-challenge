"""
Given basis vectors in two different bases B and C for R^3, write a Python function to compute the transformation matrix P from basis B to C.

Example:
Input:
B = [[1, 0, 0], 
             [0, 1, 0], 
             [0, 0, 1]]
        C = [[1, 2.3, 3], 
             [4.4, 25, 6], 
             [7.4, 8, 9]]
Output:
[[-0.6772, -0.0126, 0.2342],
                [-0.0184, 0.0505, -0.0275],
                [0.5732, -0.0345, -0.0569]]
Reasoning:
The transformation matrix P from basis B to C can be found using matrix operations involving the inverse of matrix C

"""
"""
Understanding Transformation Matrices
A transformation matrix allows us to convert the coordinates of a vector in one basis to coordinates in another basis. For bases ( B ) and ( C ) of a vector space, the transformation matrix ( P ) from ( B ) to ( C ) is calculated as follows:

Steps to Calculate the Transformation Matrix
Inverse of Basis ( C ): First, find the inverse of the matrix representing basis ( C ), denoted ( C^{-1} ).
Matrix Multiplication: Multiply ( C^{-1} ) by the matrix of basis ( B ). The result is the transformation matrix:

This matrix ( P ) can be used to transform any vector coordinates from the ( B ) basis to the ( C ) basis.
"""
import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	B = np.array(B)
	C = np.array(C)
	C_1 = np.linalg.inv(C)
	P = C_1 @ B
	return P
