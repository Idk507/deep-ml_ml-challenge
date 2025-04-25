"""
Task: Convert a Dense Matrix to Compressed Row Sparse (CSR) Format
Your task is to implement a function that converts a given dense matrix into the Compressed Row Sparse (CSR) format, an efficient storage representation for sparse matrices. The CSR format only stores non-zero elements and their positions, significantly reducing memory usage for matrices with a large number of zeros.

Write a function compressed_row_sparse_matrix(dense_matrix) that takes a 2D list dense_matrix as input and returns a tuple containing three lists:

Values array: List of all non-zero elements in row-major order.
Column indices array: Column index for each non-zero element in the values array.
Row pointer array: Cumulative number of non-zero elements per row, indicating the start of each row in the values array.
Example:
Input:
dense_matrix = [
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [3, 0, 4, 0],
    [1, 0, 0, 5]
]

vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)
print("Values array:", vals)
print("Column indices array:", col_idx)
print("Row pointer array:", row_ptr)
Output:
Values array: [1, 2, 3, 4, 1, 5]
Column indices array: [0, 1, 0, 2, 0, 3]
Row pointer array: [0, 1, 2, 4, 6]
Reasoning:
The dense matrix is converted to CSR format with the values array containing non-zero elements, column indices array storing the corresponding column index, and row pointer array indicating the start of each row in the values array.

Understanding the Compressed Row Sparse Matrix Format
The Compressed Row Sparse (CSR) format is a data-efficient representation of sparse matrices, where most of the elements are zero. This format is particularly useful in large-scale scientific computing and machine learning applications, where memory efficiency is critical.

Concepts
A sparse matrix is a matrix that contains a large number of zero elements. Storing such matrices in their full form can be inefficient, both in terms of memory and computational resources. The CSR format addresses this problem by storing only the non-zero elements and their positions in the matrix. In the CSR format, a matrix is represented by three one-dimensional arrays:

Values array: Contains all the non-zero elements of the matrix, stored row by row.
Column indices array: Stores the column index corresponding to each value in the values array.
Row pointer array: Stores the cumulative number of non-zero elements in each row, allowing quick access to each row's data.
Structure
Given a matrix:

[
1
0
0
0
0
2
0
0
3
0
4
0
1
0
0
5
]
​
  
1
0
3
1
​
  
0
2
0
0
​
  
0
0
4
0
​
  
0
0
0
5
​
  
​
 
The CSR representation would be:

Values array: [1, 2, 3, 4, 1, 5]
Column indices array: [0, 1, 0, 2, 0, 3]
Row pointer array: [0, 1, 2, 4, 6]
Explanation:
The values array holds the non-zero elements in the matrix, in row-major order.
The column indices array stores the corresponding column index of each non-zero element.
The row pointer array keeps track of where each row starts in the values array. For example, row 1 starts at index 0, row 2 starts at index 1, row 3 starts at index 2, and so on.
Applications
The CSR format is widely used in high-performance computing applications such as:

Finite element analysis (FEA)
Solving large sparse linear systems (e.g., in numerical simulations)
Machine learning algorithms (e.g., support vector machines with sparse input)
Graph-based algorithms where adjacency matrices are often sparse
The CSR format improves both memory efficiency and the speed of matrix operations by focusing only on non-zero elements.
"""
import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	vals ,col_idx , row_ptr = [],[],[0]
	for row in dense_matrix:
		for i,j in enumerate(row):
			if j!= 0 : 
				vals.append(j)
				col_idx.append(i)
		row_ptr.append(len(vals))
	return vals,col_idx,row_ptr


