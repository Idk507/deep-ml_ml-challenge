"""
Write a Python function that transforms a given matrix A using the operation 
T
−
1
A
S
T 
−1
 AS, where T and S are invertible matrices. The function should first validate if the matrices T and S are invertible, and then perform the transformation. In cases where there is no solution return -1

Example:
Input:
A = [[1, 2], [3, 4]], T = [[2, 0], [0, 2]], S = [[1, 1], [0, 1]]
Output:
[[0.5,1.5],[1.5,3.5]]
Reasoning:
The matrices T and S are used to transform matrix A by computing 
T
−
1
A
S
T 
−1
 AS.

 Matrix Transformation using 
T
−
1
A
S
T 
−1
 AS
Transforming a matrix 
A
A using the operation 
T
−
1
A
S
T 
−1
 AS involves several steps. This operation changes the basis of matrix 
A
A using two matrices 
T
T and 
S
S, with 
T
T and 
S
S being invertible, to avoid loss of information. (Multiplying by non-invertible 
S
S would result in a loss of dimensions)

Steps for Transformation
Given matrices 
A
A, 
T
T, and 
S
S:

Check Invertibility: Verify that 
T
T and 
S
S are invertible by ensuring their determinants are non-zero; otherwise, return 
−
1
−1.

Compute Inverses: Find the invers of 
T
T, denoted as 
T
−
1
T 
−1
 .

Perform Matrix Multiplication: Calculate the transformed matrix:

A
′
=
T
−
1
A
S
A 
′
 =T 
−1
 AS
Example
If:

A
=
(
1
2
3
4
)
A=( 
1
3
​
  
2
4
​
 )
T
=
(
2
0
0
2
)
T=( 
2
0
​
  
0
2
​
 )
S
=
(
1
1
0
1
)
S=( 
1
0
​
  
1
1
​
 )
Check Invertibility:
det
⁡
(
T
)
=
4
≠
0
det(T)=4

=0
det
⁡
(
S
)
=
1
≠
0
det(S)=1

=0
Compute Inverses:
T
−
1
=
(
1
2
0
0
1
2
)
T 
−1
 =( 
2
1
​
 
0
​
  
0
2
1
​
 
​
 )
Perform the Transformation:
A
′
=
T
−
1
A
S
A 
′
 =T 
−1
 AS
A
′
=
(
1
2
0
0
1
2
)
(
1
2
3
4
)
(
1
1
0
1
)
A 
′
 =( 
2
1
​
 
0
​
  
0
2
1
​
 
​
 )( 
1
3
​
  
2
4
​
 )( 
1
0
​
  
1
1
​
 )
A
′
=
(
0.5
1.5
1.5
3.5
)
A 
′
 =( 
0.5
1.5
​
  
1.5
3.5
​
 )

 """
import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	A = np.asarray(A)
    T = np.asarray(T)
    S = np.asarray(S)
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0 :
        return  -1
    T_inv = np.linalg.inv(T)
    res = T_inv @ A @ S 
    return res.tolist()
