"""
Write a Python function that multiplies a matrix by a scalar and returns the result.

Example:
Input:
matrix = [[1, 2], [3, 4]], scalar = 2
Output:
[[2, 4], [6, 8]]
Reasoning:
Each element of the matrix is multiplied by the scalar.

"""
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    return [[e*scalar for e in r]for r in matrix]
