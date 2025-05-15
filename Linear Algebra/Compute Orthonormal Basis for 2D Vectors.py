"""
Compute Orthonormal Basis for 2D Vectors

Implement a function that computes an orthonormal basis for the subspace spanned by a list of 2D vectors using the Gram-Schmidt process. The function should take a list of 2D vectors and a tolerance value (tol) to determine linear independence, returning a list of orthonormal vectors (unit length and orthogonal to each other) that span the same subspace. This is a fundamental concept in linear algebra with applications in machine learning, such as feature orthogonalization.

Example:
Input:
orthonormal_basis([[1, 0], [1, 1]])
Output:
[array([1., 0.]), array([0., 1.])]
Reasoning:
Start with [1, 0], normalize to [1, 0]. For [1, 1], subtract its projection onto [1, 0] (which is [1, 0]), leaving [0, 1]. Check if norm > 1e-10 (it is 1), then normalize to [0, 1]. The result is an orthonormal basis.

Learn About topic
Understanding the Gram-Schmidt Process
The Gram-Schmidt process transforms a set of vectors into an orthonormal basis vectors that are orthogonal (perpendicular) and have unit length for the subspace they span.

Mathematical Definition
Given vectors 
v
1
,
v
2
,
…
v 
1
​
 ,v 
2
​
 ,…, the process constructs an orthonormal set 
u
1
,
u
2
,
…
u 
1
​
 ,u 
2
​
 ,… as follows:

u
1
=
v
1
∥
v
1
∥
u 
1
​
 = 
∥v 
1
​
 ∥
v 
1
​
 
​
  (normalize the first vector).
For subsequent vectors 
v
k
v 
k
​
 :
Subtract projections: 
w
k
=
v
k
−
∑
i
=
1
k
−
1
proj
u
i
(
v
k
)
,
w 
k
​
 =v 
k
​
 −∑ 
i=1
k−1
​
 proj 
u 
i
​
 
​
 (v 
k
​
 ), where 
proj
u
i
(
v
k
)
=
(
v
k
⋅
u
i
)
u
i
proj 
u 
i
​
 
​
 (v 
k
​
 )=(v 
k
​
 ⋅u 
i
​
 )u 
i
​
 .
Normalize: 
u
k
=
w
k
∥
w
k
∥
,
u 
k
​
 = 
∥w 
k
​
 ∥
w 
k
​
 
​
 , if 
∥
w
k
∥
>
tol
∥w 
k
​
 ∥>tol.
Why Orthonormal Bases?
Orthogonal vectors simplify computations (e.g., their dot product is zero).
Unit length ensures equal scaling, useful in 
P
C
A
PCA, 
Q
R
QR decomposition, and neural network optimization.
Special Case
If a vector's norm is less than or equal to 
tol
tol (default 
1
e
−
10
1e−10), it's considered linearly dependent and excluded from the basis.

Example
For vectors [[1, 0], [1, 1]] with 
tol
=
1
e
−
10
tol=1e−10:

v
1
=
[
1
,
0
]
v 
1
​
 =[1,0], 
∥
v
1
∥
=
1
∥v 
1
​
 ∥=1, so 
u
1
=
[
1
,
0
]
u 
1
​
 =[1,0].
v
2
=
[
1
,
1
]
v 
2
​
 =[1,1], projection on 
u
1
u 
1
​
 : 
(
v
2
⋅
u
1
)
u
1
=
1
⋅
[
1
,
0
]
=
[
1
,
0
]
(v 
2
​
 ⋅u 
1
​
 )u 
1
​
 =1⋅[1,0]=[1,0].
w
2
=
[
1
,
1
]
−
[
1
,
0
]
=
[
0
,
1
]
w 
2
​
 =[1,1]−[1,0]=[0,1].
∥
w
2
∥
=
1
>
1
e
−
10
∥w 
2
​
 ∥=1>1e−10, so 
u
2
=
[
0
,
1
]
u 
2
​
 =[0,1].
Result: [[1, 0], [0, 1]], rounded to 4 decimal places.

"""
import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    basis = []
    for v in vectors : 
        v = np.array(v,dtype=float)
        for b in basis : 
            v -= np.dot(v,b)*b 
        if np.linalg.norm(v)>tol : 
            basis.append(v / np.linalg.norm(v))
    return basis
