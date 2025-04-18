"""
Task: Compute the Column Space of a Matrix
In this task, you are required to implement a function matrix_image(A) that calculates the column space of a given matrix A. The column space, also known as the image or span, consists of all linear combinations of the columns of A. To find this, you'll use concepts from linear algebra, focusing on identifying independent columns that span the matrix's image. Your task: Implement the function matrix_image(A) to return the basis vectors that span the column space of A. These vectors should be extracted from the original matrix and correspond to the independent columns.

Example:
Input:
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix_image(matrix))
Output:
# [[1, 2],
#  [4, 5],
#  [7, 8]]
Reasoning:
The column space of the matrix is spanned by the independent columns [1, 2], [4, 5], and [7, 8]. These columns form the basis vectors that represent the image of the matrix.

"""

"""
Matrix Image, Spans, and How to Calculate It
In linear algebra, the column space, also called the image or span, of a matrix is the set of all possible linear combinations of its columns. The column space gives important information about the matrix, such as the dimensions and dependencies between columns. It is useful for solving linear systems and understanding the structure of the data in the matrix. The image of a function can also be thought of as all the values the function takes in its codomain. The image of a matrix is the span of its columns - all linear combinations of its columns.

Consider the following matrix 
A
A:

A
=
[
1
2
3
4
5
6
7
8
9
]
A= 
​
  
1
4
7
​
  
2
5
8
​
  
3
6
9
​
  
​
 
The column space of 
A
A is the set of all linear combinations of its columns. In other words, any vector in the column space of 
A
A can be written as:

span
(
A
)
=
c
1
[
1
4
7
]
+
c
2
[
2
5
8
]
+
c
3
[
3
6
9
]
span(A)=c 
1
​
  
​
  
1
4
7
​
  
​
 +c 
2
​
  
​
  
2
5
8
​
  
​
 +c 
3
​
  
​
  
3
6
9
​
  
​
 
Where 
c
1
c 
1
​
 , 
c
2
c 
2
​
 , and 
c
3
c 
3
​
  are scalars representing the linear combination of the columns of matrix 
A
A.

The Image of a Matrix
The image of a matrix is spanned by its pivot columns. To find the image of a matrix, you can use the following steps:

Convert to Row Echelon Form (RREF)
The first step is to convert the matrix to its RREF using Gauss-Jordan Elimination. This finds the independent equations within the matrix. In RREF form:

Each non-zero row begins with a leading 1, called a pivot
Rows of all zeros are at the bottom of the matrix
Each leading 1 is to the right of the leading 1 in the row above
Identify Pivot Columns
Once the matrix is in RREF, the pivot columns are the columns that contain the leading 1s in each non-zero row. These columns represent the independent directions that span the column space of the matrix.

Extract Pivot Columns from the Original Matrix
Finally, to find the column space of the original matrix, you take the columns from the original matrix corresponding to the pivot columns in RREF.

Applications of Matrix Image
The matrix image has important applications in:

Solving systems of linear equations
Determining the rank of a matrix
Understanding linear transformations
It is also used in areas such as data compression, computer graphics, and signal processing to analyze and manipulate data effectively.

"""

import numpy as np

def rref(A):
    # Convert to float for division operations
    A = A.astype(np.float32)
    n, m = A.shape

    for i in range(n):
        if A[i, i] == 0:
            nonzero_current_row = np.nonzero(A[i:, i])[0] + i
            if len(nonzero_current_row) == 0:
                continue
            A[[i, nonzero_current_row[0]]] = A[[nonzero_current_row[0], i]]

        A[i] = A[i] / A[i, i]

        for j in range(n):
            if i != j:
                A[j] -= A[i] * A[j, i]
    return A

def find_pivot_columns(A):
    n, m = A.shape
    pivot_columns = []
    for i in range(n):
        nonzero = np.nonzero(A[i, :])[0]
        if len(nonzero) != 0:
            pivot_columns.append(nonzero[0])
    return pivot_columns

def matrix_image(A):
    # Find the RREF of the matrix
    Arref = rref(A)
    # Find the pivot columns
    pivot_columns = find_pivot_columns(Arref)
    # Extract the pivot columns from the original matrix
    image_basis = A[:, pivot_columns]
    return image_basis
