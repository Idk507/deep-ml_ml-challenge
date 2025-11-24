"""
Implement QR decomposition using the Gram-Schmidt process. Given a matrix A, decompose it into the product of an orthogonal matrix Q (where columns are orthonormal) and an upper triangular matrix R, such that A = Q @ R. Return both Q and R as a tuple of matrices.

Example:
Input:
A = [[3, 0], [4, 5]]
Output:
Q = [[0.6, -0.8], [0.8, 0.6]], R = [[5.0, 4.0], [0.0, 3.0]]
Reasoning:
Using Gram-Schmidt: First column of Q is normalized first column of A: 
q
1
=
1
5
[
3
,
4
]
T
=
[
0.6
,
0.8
]
T
q 
1
​
 = 
5
1
​
 [3,4] 
T
 =[0.6,0.8] 
T
 . For the second column, subtract the projection onto 
q
1
q 
1
​
  and normalize. The matrix R stores the coefficients: 
R
11
=
∣
∣
[
3
,
4
]
T
∣
∣
=
5
R 
11
​
 =∣∣[3,4] 
T
 ∣∣=5, 
R
12
=
q
1
T
[
0
,
5
]
T
=
4
R 
12
​
 =q 
1
T
​
 [0,5] 
T
 =4, and 
R
22
=
3
R 
22
​
 =3.


 Understanding QR Decomposition
QR decomposition is a fundamental matrix factorization that expresses any matrix as the product of an orthogonal matrix Q and an upper triangular matrix R. This decomposition has numerous applications in numerical linear algebra, particularly in solving linear systems and eigenvalue problems.

The Decomposition
For an 
m
×
n
m×n matrix 
A
A, QR decomposition finds:

A
=
Q
R
A=QR
Where:

Q
Q is an 
m
×
n
m×n matrix with orthonormal columns: 
Q
T
Q
=
I
Q 
T
 Q=I
R
R is an 
n
×
n
n×n upper triangular matrix
Orthonormal columns means that each column of 
Q
Q has unit length and all columns are perpendicular to each other:

q
i
T
q
j
=
{
1
if 
i
=
j
0
if 
i
≠
j
q 
i
T
​
 q 
j
​
 ={ 
1
0
​
  
if i=j
if i

=j
​
 
The Gram-Schmidt Process
The classical method for computing QR decomposition is the Gram-Schmidt orthogonalization process. It converts a set of linearly independent vectors (the columns of 
A
A) into an orthonormal set (the columns of 
Q
Q).

Step-by-step Process

For each column 
j
j of matrix 
A
A:

Start with column vector: 
v
j
=
a
j
v 
j
​
 =a 
j
​
 

Subtract projections onto previous orthonormal vectors:

v
j
=
a
j
−
∑
i
=
1
j
−
1
(
q
i
T
a
j
)
q
i
v 
j
​
 =a 
j
​
 − 
i=1
∑
j−1
​
 (q 
i
T
​
 a 
j
​
 )q 
i
​
 
This removes components of 
a
j
a 
j
​
  that lie in the directions of previously computed orthonormal vectors.

Normalize to get unit vector:
q
j
=
v
j
∣
∣
v
j
∣
∣
q 
j
​
 = 
∣∣v 
j
​
 ∣∣
v 
j
​
 
​
 
Store coefficients in R:
R
i
j
=
{
q
i
T
a
j
if 
i
<
j
∣
∣
v
j
∣
∣
if 
i
=
j
0
if 
i
>
j
R 
ij
​
 = 
⎩
⎨
⎧
​
  
q 
i
T
​
 a 
j
​
 
∣∣v 
j
​
 ∣∣
0
​
  
if i<j
if i=j
if i>j
​
 
Mathematical Foundation
Why R is Upper Triangular

The matrix 
R
R captures the relationship between the original columns 
a
j
a 
j
​
  and the orthonormal columns 
q
i
q 
i
​
 :

a
j
=
∑
i
=
1
j
R
i
j
q
i
a 
j
​
 = 
i=1
∑
j
​
 R 
ij
​
 q 
i
​
 
Since 
a
j
a 
j
​
  is expressed only in terms of 
q
1
,
…
,
q
j
q 
1
​
 ,…,q 
j
​
  (not later columns), we have 
R
i
j
=
0
R 
ij
​
 =0 for 
i
>
j
i>j, making 
R
R upper triangular.

Projection Formula

The projection of vector 
a
a onto unit vector 
q
q is:

proj
q
(
a
)
=
(
q
T
a
)
q
proj 
q
​
 (a)=(q 
T
 a)q
Gram-Schmidt repeatedly applies this to remove components in already-processed directions.

Example Calculation
Consider matrix:

A
=
(
3
0
4
5
)
A=( 
3
4
​
  
0
5
​
 )
Column 1:

v
1
=
(
3
4
)
,
∣
∣
v
1
∣
∣
=
9
+
16
=
5
v 
1
​
 =( 
3
4
​
 ),∣∣v 
1
​
 ∣∣= 
9+16
​
 =5
q
1
=
1
5
(
3
4
)
=
(
0.6
0.8
)
q 
1
​
 = 
5
1
​
 ( 
3
4
​
 )=( 
0.6
0.8
​
 )
R
11
=
5
R 
11
​
 =5
Column 2:

Projection of 
[
0
,
5
]
T
[0,5] 
T
  onto 
q
1
q 
1
​
 :

R
12
=
q
1
T
a
2
=
0.6
(
0
)
+
0.8
(
5
)
=
4
R 
12
​
 =q 
1
T
​
 a 
2
​
 =0.6(0)+0.8(5)=4
Remove this projection:

v
2
=
(
0
5
)
−
4
(
0.6
0.8
)
=
(
−
2.4
1.8
)
v 
2
​
 =( 
0
5
​
 )−4( 
0.6
0.8
​
 )=( 
−2.4
1.8
​
 )
∣
∣
v
2
∣
∣
=
2.4
2
+
1.8
2
=
3
,
R
22
=
3
∣∣v 
2
​
 ∣∣= 
2.4 
2
 +1.8 
2
 
​
 =3,R 
22
​
 =3
q
2
=
1
3
(
−
2.4
1.8
)
=
(
−
0.8
0.6
)
q 
2
​
 = 
3
1
​
 ( 
−2.4
1.8
​
 )=( 
−0.8
0.6
​
 )
Result:

Q
=
(
0.6
−
0.8
0.8
0.6
)
,
R
=
(
5
4
0
3
)
Q=( 
0.6
0.8
​
  
−0.8
0.6
​
 ),R=( 
5
0
​
  
4
3
​
 )
Properties and Verification
Orthogonality of Q

Verify that 
Q
T
Q
=
I
Q 
T
 Q=I:

Q
T
Q
=
(
0.6
0.8
−
0.8
0.6
)
(
0.6
−
0.8
0.8
0.6
)
=
(
1
0
0
1
)
Q 
T
 Q=( 
0.6
−0.8
​
  
0.8
0.6
​
 )( 
0.6
0.8
​
  
−0.8
0.6
​
 )=( 
1
0
​
  
0
1
​
 )
Reconstruction

Verify that 
Q
R
=
A
QR=A:

Q
R
=
(
0.6
−
0.8
0.8
0.6
)
(
5
4
0
3
)
=
(
3
0
4
5
)
=
A
QR=( 
0.6
0.8
​
  
−0.8
0.6
​
 )( 
5
0
​
  
4
3
​
 )=( 
3
4
​
  
0
5
​
 )=A
Uniqueness and Sign Ambiguity
QR decomposition is not unique: you can multiply any column of 
Q
Q by 
−
1
−1 and the corresponding row of 
R
R by 
−
1
−1, and the product remains unchanged. Different implementations may produce different valid decompositions.

Applications
Solving Linear Systems: For 
A
x
=
b
Ax=b, substitute 
A
=
Q
R
A=QR:

Q
R
x
=
b
  
⟹
  
R
x
=
Q
T
b
QRx=b⟹Rx=Q 
T
 b
Since 
R
R is upper triangular, this can be solved efficiently by back-substitution.

Least Squares: For overdetermined systems (
m
>
n
m>n), QR decomposition provides a stable way to solve 
min
⁡
x
∣
∣
A
x
−
b
∣
∣
2
min 
x
​
 ∣∣Ax−b∣∣ 
2
 .

Eigenvalue Algorithms: The QR algorithm iteratively applies QR decomposition to find eigenvalues.

Orthogonalization: Converting any basis into an orthonormal basis.

Computational Considerations
The classical Gram-Schmidt process can be numerically unstable due to rounding errors. Modified Gram-Schmidt improves stability by reorthogonalizing at each step. For large matrices, Householder reflections or Givens rotations are preferred for their superior numerical properties.

Time complexity: 
O
(
m
n
2
)
O(mn 
2
 ) for an 
m
×
n
m×n matrix.


"""
import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix represented as list of lists
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    # Convert input to numpy array for easier math
    A = np.array(A, dtype=float)
    m, n = A.shape
    
    # Initialize Q and R
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        
        # Subtract projections onto previous q_i
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        
        # Compute norm and normalize
        R[j, j] = np.linalg.norm(v)
        if R[j, j] == 0:
            raise ValueError("Matrix has linearly dependent columns; QR decomposition not possible.")
        Q[:, j] = v / R[j, j]
    
    return Q.tolist(), R.tolist()
