"""
Write a Python function called svd_2x2_singular_values(A) that finds an approximate singular value decomposition of a real 2 x 2 matrix using one Jacobi rotation. Input A: a NumPy array of shape (2, 2)

Rules You may use basic NumPy operations (matrix multiplication, transpose, element-wise math, etc.). Do not call numpy.linalg.svd or any other high-level SVD routine. Stick to a single Jacobi stepâno iterative refinements.

Return A tuple (U, Î£, V_T) where U is a 2 x 2 orthogonal matrix, Î£ is a length-2 NumPy array containing the singular values, and V_T is the transpose of the right-singular-vector matrix V.

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
    
