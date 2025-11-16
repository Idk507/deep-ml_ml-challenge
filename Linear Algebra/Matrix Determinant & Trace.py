"""
Implement a function that computes both the determinant and trace of a square matrix. The determinant is a scalar value that can be computed from the elements of a square matrix and encodes certain properties of the matrix. The trace is simply the sum of the elements on the main diagonal. Return both values as a tuple.

Example:
Input:
matrix=[[2, 3], [1, 4]]
Output:
(5.0, 6.0)
Reasoning:
The determinant of a 2x2 matrix is computed as 
a
d
−
b
c
ad−bc where the matrix is 
(
a
b
c
d
)
( 
a
c
​
  
b
d
​
 ). Here: 
2
×
4
−
3
×
1
=
8
−
3
=
5
2×4−3×1=8−3=5. The trace is the sum of diagonal elements: 
2
+
4
=
6
2+4=6.

Understanding Matrix Determinant and Trace
The determinant and trace are two fundamental scalar values that can be computed from square matrices. Each provides important information about the matrix's properties and behavior in linear transformations.

The Trace
The trace is the simplest of the two concepts: it's the sum of the diagonal elements.

Definition

For an 
n
×
n
n×n matrix 
A
A:

tr
(
A
)
=
∑
i
=
1
n
a
i
i
tr(A)= 
i=1
∑
n
​
 a 
ii
​
 
For example, for matrix:

A
=
(
2
3
1
0
5
4
7
6
8
)
A= 
​
  
2
0
7
​
  
3
5
6
​
  
1
4
8
​
  
​
 
The trace is: 
tr
(
A
)
=
2
+
5
+
8
=
15
tr(A)=2+5+8=15

Properties of Trace

Linearity: 
tr
(
A
+
B
)
=
tr
(
A
)
+
tr
(
B
)
tr(A+B)=tr(A)+tr(B) and 
tr
(
c
A
)
=
c
⋅
tr
(
A
)
tr(cA)=c⋅tr(A)
Cyclic property: 
tr
(
A
B
C
)
=
tr
(
C
A
B
)
=
tr
(
B
C
A
)
tr(ABC)=tr(CAB)=tr(BCA)
Invariant under transpose: 
tr
(
A
T
)
=
tr
(
A
)
tr(A 
T
 )=tr(A)
Sum of eigenvalues: 
tr
(
A
)
=
∑
i
=
1
n
λ
i
tr(A)=∑ 
i=1
n
​
 λ 
i
​
  where 
λ
i
λ 
i
​
  are eigenvalues
The Determinant
The determinant is a scalar value that encodes information about the matrix's invertibility, the volume scaling of the linear transformation it represents, and other properties.

Definition for 2Ã2 Matrices

For a 2Ã2 matrix, the determinant has a simple formula:

det
⁡
(
a
b
c
d
)
=
a
d
−
b
c
det( 
a
c
​
  
b
d
​
 )=ad−bc
Cofactor Expansion for Larger Matrices

For 
n
×
n
n×n matrices where 
n
>
2
n>2, the determinant can be computed recursively using cofactor expansion along any row or column. Expanding along row 
i
i:

det
⁡
(
A
)
=
∑
j
=
1
n
(
−
1
)
i
+
j
a
i
j
M
i
j
det(A)= 
j=1
∑
n
​
 (−1) 
i+j
 a 
ij
​
 M 
ij
​
 
Where:

a
i
j
a 
ij
​
  is the element at row 
i
i, column 
j
j
M
i
j
M 
ij
​
  is the determinant of the 
(
n
−
1
)
×
(
n
−
1
)
(n−1)×(n−1) minor matrix obtained by removing row 
i
i and column 
j
j
(
−
1
)
i
+
j
(−1) 
i+j
  provides the alternating sign pattern
Example: 3Ã3 Determinant

For matrix:

A
=
(
1
2
3
0
1
4
5
6
0
)
A= 
​
  
1
0
5
​
  
2
1
6
​
  
3
4
0
​
  
​
 
Expanding along the first row:

det
⁡
(
A
)
=
1
⋅
∣
1
4
6
0
∣
−
2
⋅
∣
0
4
5
0
∣
+
3
⋅
∣
0
1
5
6
∣
det(A)=1⋅ 
​
  
1
6
​
  
4
0
​
  
​
 −2⋅ 
​
  
0
5
​
  
4
0
​
  
​
 +3⋅ 
​
  
0
5
​
  
1
6
​
  
​
 
=
1
(
1
⋅
0
−
4
⋅
6
)
−
2
(
0
⋅
0
−
4
⋅
5
)
+
3
(
0
⋅
6
−
1
⋅
5
)
=1(1⋅0−4⋅6)−2(0⋅0−4⋅5)+3(0⋅6−1⋅5)
=
1
(
−
24
)
−
2
(
−
20
)
+
3
(
−
5
)
=
−
24
+
40
−
15
=
1
=1(−24)−2(−20)+3(−5)=−24+40−15=1
Properties of Determinant
Invertibility: A matrix is invertible if and only if 
det
⁡
(
A
)
≠
0
det(A)

=0. Matrices with zero determinant are called singular.

Multiplicativity: 
det
⁡
(
A
B
)
=
det
⁡
(
A
)
⋅
det
⁡
(
B
)
det(AB)=det(A)⋅det(B)

Transpose: 
det
⁡
(
A
T
)
=
det
⁡
(
A
)
det(A 
T
 )=det(A)

Inverse: If 
A
A is invertible, 
det
⁡
(
A
−
1
)
=
1
det
⁡
(
A
)
det(A 
−1
 )= 
det(A)
1
​
 

Row operations:

Swapping two rows multiplies the determinant by 
−
1
−1
Multiplying a row by scalar 
c
c multiplies the determinant by 
c
c
Adding a multiple of one row to another doesn't change the determinant
Product of eigenvalues: 
det
⁡
(
A
)
=
∏
i
=
1
n
λ
i
det(A)=∏ 
i=1
n
​
 λ 
i
​
  where 
λ
i
λ 
i
​
  are eigenvalues

Geometric Interpretation
Determinant as Volume

The absolute value of the determinant represents the volume scaling factor of the linear transformation. If 
T
T is the transformation represented by matrix 
A
A:

Volume
(
T
(
S
)
)
=
∣
det
⁡
(
A
)
∣
⋅
Volume
(
S
)
Volume(T(S))=∣det(A)∣⋅Volume(S)
For a 2D transformation, 
∣
det
⁡
(
A
)
∣
∣det(A)∣ gives the area scaling factor. For 3D, it gives the volume scaling factor.

Sign of Determinant

det
⁡
(
A
)
>
0
det(A)>0: Transformation preserves orientation
det
⁡
(
A
)
<
0
det(A)<0: Transformation reverses orientation
det
⁡
(
A
)
=
0
det(A)=0: Transformation collapses space into lower dimension
Computational Considerations
Cofactor expansion has 
O
(
n
!
)
O(n!) time complexity, making it impractical for large matrices. For efficient computation:

LU Decomposition: Decompose 
A
=
L
U
A=LU where 
L
L is lower triangular and 
U
U is upper triangular. Then:

det
⁡
(
A
)
=
det
⁡
(
L
)
⋅
det
⁡
(
U
)
=
∏
i
=
1
n
u
i
i
det(A)=det(L)⋅det(U)= 
i=1
∏
n
​
 u 
ii
​
 
This reduces complexity to 
O
(
n
3
)
O(n 
3
 ).

For trace: Direct computation is always 
O
(
n
)
O(n)âsimply sum the diagonal.

Relationship Between Determinant and Trace
For a 2Ã2 matrix with eigenvalues 
λ
1
λ 
1
​
  and 
λ
2
λ 
2
​
 :

tr
(
A
)
=
λ
1
+
λ
2
tr(A)=λ 
1
​
 +λ 
2
​
 
det
⁡
(
A
)
=
λ
1
⋅
λ
2
det(A)=λ 
1
​
 ⋅λ 
2
​
 
These two values provide enough information to find the eigenvalues using the characteristic equation:

λ
2
−
tr
(
A
)
⋅
λ
+
det
⁡
(
A
)
=
0
λ 
2
 −tr(A)⋅λ+det(A)=0
Applications
Computer Graphics: Determinants determine if transformations preserve or flip orientation, crucial for backface culling.

Machine Learning: The determinant appears in multivariate Gaussian distributions and covariance matrices.

Differential Equations: The trace and determinant of the coefficient matrix determine the stability and behavior of linear systems.

Optimization: The Hessian's determinant indicates whether a critical point is a minimum, maximum, or saddle point.
"""

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.

    Args:
        matrix: A square matrix (n x n) represented as list of lists

    Returns:
        Tuple of (determinant, trace)
    """
    def minor(mat, i, j):
        # Return the minor matrix after removing row i and column j
        return [row[:j] + row[j+1:] for idx, row in enumerate(mat) if idx != i]

    def determinant(mat):
        n = len(mat)
        if n == 1:
            return mat[0][0]
        if n == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        det = 0
        for j in range(n):
            cofactor = (-1) ** j * mat[0][j] * determinant(minor(mat, 0, j))
            det += cofactor
        return det

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Input must be a square matrix")

    trace = sum(matrix[i][i] for i in range(n))
    det = determinant(matrix)

    return float(det), float(trace)

