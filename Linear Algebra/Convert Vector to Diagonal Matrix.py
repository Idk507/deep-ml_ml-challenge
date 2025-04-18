"""
Write a Python function to convert a 1D numpy array into a diagonal matrix. The function should take in a 1D numpy array x and return a 2D numpy array representing the diagonal matrix.

Example:
Input:
x = np.array([1, 2, 3])
    output = make_diagonal(x)
    print(output)
Output:
[[1. 0. 0.]
    [0. 2. 0.]
    [0. 0. 3.]]
Reasoning:
The input vector [1, 2, 3] is converted into a diagonal matrix where the elements of the vector form the diagonal of the matrix.
Understanding Diagonal Matrices
A diagonal matrix is a square matrix in which the entries outside the main diagonal are all zero. The main diagonal is the set of entries extending from the top left to the bottom right of the matrix.

Problem Overview
In this problem, you will write a function to convert a 1D numpy array (vector) into a diagonal matrix. The resulting matrix will have the elements of the input vector on its main diagonal, with zeros elsewhere.

Mathematical Representation
Given a vector 

  
​
 
Importance
Diagonal matrices are important in various mathematical and scientific computations due to their simple structure and useful properties.
"""
import numpy as np

def make_diagonal(x):
	# Your code here
	return np.diag(x)
