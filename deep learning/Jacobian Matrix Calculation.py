"""
Jacobian Matrix Calculation
Medium
Calculus

Implement a function to compute the Jacobian matrix of a vector-valued function using numerical differentiation. The Jacobian matrix contains all first-order partial derivatives of a function f: R^n -> R^m. Given a function f and a point x, approximate each partial derivative using finite differences and return the m x n Jacobian matrix.

Example:
Input:
f(x, y) = [x^2, xy, y^2], x = [2, 3]
Output:
[[4.0, 0.0], [3.0, 2.0], [0.0, 6.0]]
Reasoning:
The Jacobian is a 3x2 matrix. First row: 
∂
(
x
2
)
∂
x
=
2
x
=
4
∂x
∂(x 
2
 )
​
 =2x=4, 
∂
(
x
2
)
∂
y
=
0
∂y
∂(x 
2
 )
​
 =0. Second row: 
∂
(
x
y
)
∂
x
=
y
=
3
∂x
∂(xy)
​
 =y=3, 
∂
(
x
y
)
∂
y
=
x
=
2
∂y
∂(xy)
​
 =x=2. Third row: 
∂
(
y
2
)
∂
x
=
0
∂x
∂(y 
2
 )
​
 =0, 
∂
(
y
2
)
∂
y
=
2
y
=
6
∂y
∂(y 
2
 )
​
 =2y=6.

Learn About topic
Understanding the Jacobian Matrix
The Jacobian matrix is a fundamental concept in multivariable calculus that generalizes the derivative to vector-valued functions. It captures how a function transforms space locally and is essential in optimization, robotics, and numerical methods.

Definition
For a function 
f
:
R
n
→
R
m
f:R 
n
 →R 
m
  that maps an n-dimensional input to an m-dimensional output:

f
(
x
)
=
(
f
1
(
x
1
,
…
,
x
n
)
f
2
(
x
1
,
…
,
x
n
)
⋮
f
m
(
x
1
,
…
,
x
n
)
)
f(x)= 
​
  
f 
1
​
 (x 
1
​
 ,…,x 
n
​
 )
f 
2
​
 (x 
1
​
 ,…,x 
n
​
 )
⋮
f 
m
​
 (x 
1
​
 ,…,x 
n
​
 )
​
  
​
 
The Jacobian matrix 
J
J is the 
m
×
n
m×n matrix of all first-order partial derivatives:

J
=
(
∂
f
1
∂
x
1
∂
f
1
∂
x
2
⋯
∂
f
1
∂
x
n
∂
f
2
∂
x
1
∂
f
2
∂
x
2
⋯
∂
f
2
∂
x
n
⋮
⋮
⋱
⋮
∂
f
m
∂
x
1
∂
f
m
∂
x
2
⋯
∂
f
m
∂
x
n
)
J= 
​
  
∂x 
1
​
 
∂f 
1
​
 
​
 
∂x 
1
​
 
∂f 
2
​
 
​
 
⋮
∂x 
1
​
 
∂f 
m
​
 
​
 
​
  
∂x 
2
​
 
∂f 
1
​
 
​
 
∂x 
2
​
 
∂f 
2
​
 
​
 
⋮
∂x 
2
​
 
∂f 
m
​
 
​
 
​
  
⋯
⋯
⋱
⋯
​
  
∂x 
n
​
 
∂f 
1
​
 
​
 
∂x 
n
​
 
∂f 
2
​
 
​
 
⋮
∂x 
n
​
 
∂f 
m
​
 
​
 
​
  
​
 
Each entry 
J
i
j
J 
ij
​
  represents:

J
i
j
=
∂
f
i
∂
x
j
J 
ij
​
 = 
∂x 
j
​
 
∂f 
i
​
 
​
 
This tells us how the 
i
i-th output component changes with respect to the 
j
j-th input component.

Geometric Interpretation
The Jacobian matrix represents the best linear approximation of a function near a point. For small displacements 
Δ
x
Δx from point 
x
x:

f
(
x
+
Δ
x
)
≈
f
(
x
)
+
J
(
x
)
Δ
x
f(x+Δx)≈f(x)+J(x)Δx
This is the multivariate generalization of the linear approximation 
f
(
x
+
Δ
x
)
≈
f
(
x
)
+
f
′
(
x
)
Δ
x
f(x+Δx)≈f(x)+f 
′
 (x)Δx.

Example: Quadratic Functions
Consider 
f
(
x
,
y
)
=
[
x
2
,
x
y
,
y
2
]
f(x,y)=[x 
2
 ,xy,y 
2
 ] evaluated at point 
(
2
,
3
)
(2,3).

Computing partial derivatives:

∂
f
1
∂
x
=
∂
(
x
2
)
∂
x
=
2
x
∣
(
2
,
3
)
=
4
∂x
∂f 
1
​
 
​
 = 
∂x
∂(x 
2
 )
​
 =2x 
​
  
(2,3)
​
 =4
∂
f
1
∂
y
=
∂
(
x
2
)
∂
y
=
0
∂y
∂f 
1
​
 
​
 = 
∂y
∂(x 
2
 )
​
 =0
∂
f
2
∂
x
=
∂
(
x
y
)
∂
x
=
y
∣
(
2
,
3
)
=
3
∂x
∂f 
2
​
 
​
 = 
∂x
∂(xy)
​
 =y 
​
  
(2,3)
​
 =3
∂
f
2
∂
y
=
∂
(
x
y
)
∂
y
=
x
∣
(
2
,
3
)
=
2
∂y
∂f 
2
​
 
​
 = 
∂y
∂(xy)
​
 =x 
​
  
(2,3)
​
 =2
∂
f
3
∂
x
=
∂
(
y
2
)
∂
x
=
0
∂x
∂f 
3
​
 
​
 = 
∂x
∂(y 
2
 )
​
 =0
∂
f
3
∂
y
=
∂
(
y
2
)
∂
y
=
2
y
∣
(
2
,
3
)
=
6
∂y
∂f 
3
​
 
​
 = 
∂y
∂(y 
2
 )
​
 =2y 
​
  
(2,3)
​
 =6
Jacobian matrix:

J
=
(
4
0
3
2
0
6
)
J= 
​
  
4
3
0
​
  
0
2
6
​
  
​
 
Numerical Approximation
When analytical derivatives are unavailable or impractical, we use finite differences to approximate partial derivatives:

∂
f
i
∂
x
j
≈
f
i
(
x
1
,
…
,
x
j
+
h
,
…
,
x
n
)
−
f
i
(
x
1
,
…
,
x
j
,
…
,
x
n
)
h
∂x 
j
​
 
∂f 
i
​
 
​
 ≈ 
h
f 
i
​
 (x 
1
​
 ,…,x 
j
​
 +h,…,x 
n
​
 )−f 
i
​
 (x 
1
​
 ,…,x 
j
​
 ,…,x 
n
​
 )
​
 
This is called the forward difference approximation. For small 
h
h (typically 
10
−
5
10 
−5
  to 
10
−
8
10 
−8
 ), this approximates the true derivative with error 
O
(
h
)
O(h).

Central difference (more accurate but requires more function evaluations):

∂
f
i
∂
x
j
≈
f
i
(
…
,
x
j
+
h
,
…
)
−
f
i
(
…
,
x
j
−
h
,
…
)
2
h
∂x 
j
​
 
∂f 
i
​
 
​
 ≈ 
2h
f 
i
​
 (…,x 
j
​
 +h,…)−f 
i
​
 (…,x 
j
​
 −h,…)
​
 
This has error 
O
(
h
2
)
O(h 
2
 ), providing better accuracy for the same step size.

Special Cases
Gradient (n inputs, 1 output):

When 
m
=
1
m=1, the Jacobian is a 
1
×
n
1×n row vector, which is the transpose of the gradient:

J
=
∇
f
T
=
(
∂
f
∂
x
1
∂
f
∂
x
2
⋯
∂
f
∂
x
n
)
J=∇f 
T
 =( 
∂x 
1
​
 
∂f
​
 
​
  
∂x 
2
​
 
∂f
​
 
​
  
⋯
​
  
∂x 
n
​
 
∂f
​
 
​
 )
Derivative (1 input, m outputs):

When 
n
=
1
n=1, the Jacobian is an 
m
×
1
m×1 column vector:

J
=
(
d
f
1
d
x
d
f
2
d
x
⋮
d
f
m
d
x
)
J= 
​
  
dx
df 
1
​
 
​
 
dx
df 
2
​
 
​
 
⋮
dx
df 
m
​
 
​
 
​
  
​
 
Scalar derivative (1 input, 1 output):

When both 
m
=
1
m=1 and 
n
=
1
n=1, the Jacobian reduces to the ordinary derivative: 
J
=
f
′
(
x
)
J=f 
′
 (x).

The Jacobian Determinant
When the Jacobian is square (
m
=
n
m=n), its determinant 
det
⁡
(
J
)
det(J) has important meaning:

Volume scaling: 
∣
det
⁡
(
J
)
∣
∣det(J)∣ represents the local volume scaling factor of the transformation
Invertibility: 
det
⁡
(
J
)
≠
0
det(J)

=0 means the function is locally invertible near that point (Inverse Function Theorem)
Change of variables: In integration, 
det
⁡
(
J
)
det(J) appears in change-of-variable formulas:
∫
f
(
D
)
g
(
y
)
d
y
=
∫
D
g
(
f
(
x
)
)
∣
det
⁡
(
J
(
x
)
)
∣
d
x
∫ 
f(D)
​
 g(y)dy=∫ 
D
​
 g(f(x))∣det(J(x))∣dx
Applications
Newton's Method for Systems: To solve 
f
(
x
)
=
0
f(x)=0, iterate:

x
k
+
1
=
x
k
−
J
(
x
k
)
−
1
f
(
x
k
)
x 
k+1
​
 =x 
k
​
 −J(x 
k
​
 ) 
−1
 f(x 
k
​
 )
Backpropagation in Neural Networks: The Jacobian represents how network outputs change with respect to parameters, enabling gradient-based optimization via the chain rule.

Robotics: The Jacobian maps joint velocities to end-effector velocities in manipulators:

v
e
n
d
=
J
(
q
)
q
˙
v 
end
​
 =J(q) 
q
˙
​
 
Where 
q
q are joint angles and 
v
e
n
d
v 
end
​
  is end-effector velocity.

Sensitivity Analysis: The Jacobian quantifies how sensitive outputs are to input perturbations, crucial for understanding model behavior and uncertainty propagation.

The Chain Rule
For composite functions 
h
(
x
)
=
g
(
f
(
x
)
)
h(x)=g(f(x)), the chain rule states:

J
h
(
x
)
=
J
g
(
f
(
x
)
)
⋅
J
f
(
x
)
J 
h
​
 (x)=J 
g
​
 (f(x))⋅J 
f
​
 (x)
This matrix multiplication is the foundation of automatic differentiation and backpropagation algorithms.

Computational Considerations
Computing the Jacobian numerically requires 
n
n function evaluations (one per column). For functions 
R
n
→
R
m
R 
n
 →R 
m
 , this gives complexity:

Time: 
O
(
n
⋅
T
f
)
O(n⋅T 
f
​
 ) where 
T
f
T 
f
​
  is the time to evaluate 
f
f
Space: 
O
(
m
n
)
O(mn) to store the Jacobian
For large 
n
n, automatic differentiation techniques can compute Jacobian-vector products more efficiently than constructing the full matrix.

"""
import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix as list of lists
    """
    # Convert x to numpy array for easier manipulation
    x_array = np.array(x, dtype=np.float64)
    n = len(x_array)  # Number of inputs
    
    # Evaluate function at the point x to determine output dimension
    f_x = np.array(f(x_array.tolist()), dtype=np.float64)
    m = len(f_x)  # Number of outputs
    
    # Initialize Jacobian matrix (m x n)
    jacobian = np.zeros((m, n))
    
    # Compute each column of the Jacobian
    # Column j corresponds to partial derivatives with respect to x_j
    for j in range(n):
        # Create perturbed point: x + h*e_j where e_j is the j-th unit vector
        x_perturbed = x_array.copy()
        x_perturbed[j] += h
        
        # Evaluate function at perturbed point
        f_perturbed = np.array(f(x_perturbed.tolist()), dtype=np.float64)
        
        # Compute finite difference approximation: (f(x + h*e_j) - f(x)) / h
        # This gives us all partial derivatives ∂f_i/∂x_j for i = 1, ..., m
        jacobian[:, j] = (f_perturbed - f_x) / h
    
    # Convert to list of lists
    return jacobian.tolist()
