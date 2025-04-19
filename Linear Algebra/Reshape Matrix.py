"""
Write a Python function that reshapes a given matrix into a specified shape. if it cant be reshaped return back an empty list [ ]

Example:
Input:
a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)
Output:
[[1, 2], [3, 4], [5, 6], [7, 8]]
Reasoning:
The given matrix is reshaped from 2x4 to 4x2.

"""
"""
Reshaping a Matrix
Matrix reshaping involves changing the shape of a matrix without altering its data. This is essential in many machine learning tasks where the input data needs to be formatted in a specific way.

For example, consider a matrix 
M
M:

Original Matrix 
M
M:

M
=
(
1
2
3
4
5
6
7
8
)
M=( 
1
5
​
  
2
6
​
  
3
7
​
  
4
8
​
 )
Reshaped Matrix 
M
′
M 
′
  with shape (4, 2):

M
′
=
(
1
2
3
4
5
6
7
8
)
M 
′
 = 
​
  
1
3
5
7
​
  
2
4
6
8
​
  
​
 
Important Note:
Ensure the total number of elements remains constant during reshaping.

"""
import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int|float]) -> list[list[int|float]]:
    # Not compatible case
    if len(a)*len(a[0]) != new_shape[0]*new_shape[1]:
        return []
    return np.array(a).reshape(new_shape).tolist()
