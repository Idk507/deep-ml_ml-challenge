"""
Calculate 2x2 Matrix Inverse

Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.

Example:
Input:
matrix = [[4, 7], [2, 6]]
Output:
[[0.6, -0.7], [-0.2, 0.4]]
Reasoning:
The inverse of a 2x2 matrix [a, b], [c, d] is given by (1/(ad-bc)) * [d, -b], [-c, a], provided ad-bc is not zero.

"""
import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
	return np.linalg.inv(matrix)
