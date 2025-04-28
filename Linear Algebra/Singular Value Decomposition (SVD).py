"""
Write a Python function that approximates the Singular Value Decomposition on a 2x2 matrix by using the jacobian method and without using numpy svd function, i mean you could but you wouldn't learn anything. return the result in this format.

Example:
Input:
a = [[2, 1], [1, 2]]
Output:
(array([[-0.70710678, -0.70710678],
                        [-0.70710678,  0.70710678]]),
        array([3., 1.]),
        array([[-0.70710678, -0.70710678],
               [-0.70710678,  0.70710678]]))
Reasoning:
U is the first matrix sigma is the second vector and V is the third matrix


Singular Value Decomposition (SVD) via the Jacobi Method
Singular Value Decomposition (SVD) is a powerful matrix decomposition technique in linear algebra that expresses a matrix as the product of three other matrices, revealing its intrinsic geometric and algebraic properties. When using the Jacobi method, SVD decomposes a matrix ( A ) into:

A
=
U
Σ
V
T
A=UΣV 
T
 
( A ) is the original ( m \times n ) matrix.
( U ) is an ( m \times m ) orthogonal matrix whose columns are the left singular vectors of ( A ).
( \Sigma ) is an ( m \times n ) diagonal matrix containing the singular values of ( A ).
( V^T ) is the transpose of an ( n \times n ) orthogonal matrix whose columns are the right singular vectors of ( A ).
The Jacobi Method for SVD
The Jacobi method is an iterative algorithm used for diagonalizing a symmetric matrix through a series of rotational transformations. It is particularly suited for computing the SVD by iteratively applying rotations to minimize off-diagonal elements until the matrix is diagonal.

Steps of the Jacobi SVD Algorithm
Initialization: Start with ( A^TA ) (or ( AA^T ) for ( U )) and set ( V ) (or ( U )) as an identity matrix. The goal is to diagonalize ( A^TA ), obtaining ( V ) in the process.
Choosing Rotation Targets: Identify off-diagonal elements in ( A^TA ) to be minimized or zeroed out through rotations.
Calculating Rotation Angles: For each target off-diagonal element, calculate the angle ( \theta ) for the Jacobi rotation matrix ( J ) that would zero it. This involves solving for ( \theta ) using (\text{atan2}) to accurately handle the quadrant of rotation:
θ
=
0.5
⋅
atan2
(
2
a
i
j
,
a
i
i
−
a
j
j
)
θ=0.5⋅atan2(2a 
ij
​
 ,a 
ii
​
 −a 
jj
​
 )
where ( a_{ij} ) is the target off-diagonal element, and ( a_{ii} ), ( a_{jj} ) are the diagonal elements of ( A^TA ).
Applying Rotations: Construct ( J ) using ( \theta ) and apply the rotation to ( A^TA ), effectively reducing the magnitude of the target off-diagonal element. Update ( V ) (or ( U )) by multiplying it by ( J ).
Iteration and Convergence: Repeat the process of selecting off-diagonal elements, calculating rotation angles, and applying rotations until ( A^TA ) is sufficiently diagonalized.
Extracting SVD Components: Once diagonalized, the diagonal entries of ( A^TA ) represent the squared singular values of ( A ). The matrices ( U ) and ( V ) are constructed from the accumulated rotations, containing the left and right singular vectors of ( A ), respectively.
Practical Considerations
The Jacobi method is particularly effective for dense matrices where off-diagonal elements are significant.
Careful implementation is required to ensure numerical stability and efficiency, especially for large matrices.
The iterative nature of the Jacobi method makes it computationally intensive, but it is highly parallelizable.

"""

import numpy as np 


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
   # stick to lowercase
   a = A

   a_t = np.transpose(a)
   a_2 = a_t @ a

   v = np.eye(2)

   for _ in range(1):
       # Compute rotation angle for a 2x2 matrix
       if a_2[0,0] == a_2[1,1]:
           theta = np.pi/4
       else:
           theta = 0.5 * np.arctan2(2 * a_2[0,1], a_2[0,0] - a_2[1,1])
       
       # Create rotation matrix
       r = np.array(
           [
               [np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]
               ]
           )
       
       # apply rotation
       d = np.transpose(r) @ a_2 @ r

       # update a_2
       a_2 = d

       # accumulate v
       v = v @ r

   # sigma is the diagonal elements squared
   s = np.sqrt([d[0,0], d[1,1]])
   s_inv = np.array([[1/s[0], 0], [0, 1/s[1]]])
   
   u = a @ v @ s_inv
   
   return (u, s, v.T)
    
