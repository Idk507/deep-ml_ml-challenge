"""
Task: Create a Compressed Column Sparse Matrix Representation
Your task is to implement a function that converts a dense matrix into its Compressed Column Sparse (CSC) representation. The CSC format stores only non-zero elements of the matrix and is efficient for matrices with a high number of zero elements.

Write a function compressed_col_sparse_matrix(dense_matrix) that takes in a two-dimensional list dense_matrix and returns a tuple of three lists:

values: List of non-zero elements, stored in column-major order.
row indices: List of row indices corresponding to each value in the values array.
column pointer: List that indicates the starting index of each column in the values array.
Example:
Input:
dense_matrix = [
    [0, 0, 3, 0],
    [1, 0, 0, 4],
    [0, 2, 0, 0]
]

vals, row_idx, col_ptr = compressed_col_sparse_matrix(dense_matrix)
Output:
[1, 2, 3, 4] [1, 2, 0, 1] [0, 1, 2, 3, 4]
Reasoning:
The dense matrix is converted to CSC format with the values array containing non-zero elements, row indices array storing the corresponding row index, and column pointer array indicating the start of each column in the values array.



"""
def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	val = []
    col_ptr = [0]
    row_idx = []
    rows,cols = len(dense_matrix),len(dense_matrix)

    for i in range(len(dense_matrix)):
        for j in range(len(dense_matrix)):
            v = dense_matrix[j][i]
            if v!= 0 :
                val.append(v)
                row_idx.append(j)
        col_ptr.append(len(val))
    
    return val,col_ptr,row_idx
