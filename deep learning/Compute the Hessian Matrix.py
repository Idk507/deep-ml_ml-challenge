"""
Write a Python function that computes the Hessian matrix of a scalar-valued function at a given point. The Hessian matrix contains all second-order partial derivatives and is essential for optimization algorithms like Newton's method, as well as for analyzing the curvature of loss surfaces in machine learning. Your function should use finite differences to numerically approximate the Hessian for any arbitrary function.

Example:
Input:
f(x, y) = x^2 + y^2, point = [0.0, 0.0]
Output:
[[2.0, 0.0], [0.0, 2.0]]
Reasoning:
For f(x,y) = x^2 + y^2: The second derivative with respect to x is 2, the second derivative with respect to y is 2, and the mixed partial derivatives are 0. This gives us H = [[2, 0], [0, 2]], which indicates the function has positive curvature in all directions (it's a paraboloid).

The Hessian Matrix
Definition
The Hessian matrix is the square matrix of second-order partial derivatives of a scalar-valued function. For a function 
f
:
R
n
→
R
f:R 
n
 →R, the Hessian 
H
H is an 
n
×
n
n×n matrix:

H
=
(
∂
2
f
∂
x
1
2
∂
2
f
∂
x
1
∂
x
2
⋯
∂
2
f
∂
x
1
∂
x
n
∂
2
f
∂
x
2
∂
x
1
∂
2
f
∂
x
2
2
⋯
∂
2
f
∂
x
2
∂
x
n
⋮
⋮
⋱
⋮
∂
2
f
∂
x
n
∂
x
1
∂
2
f
∂
x
n
∂
x
2
⋯
∂
2
f
∂
x
n
2
)
H= 
​
  
∂x 
1
2
​
 
∂ 
2
 f
​
 
∂x 
2
​
 ∂x 
1
​
 
∂ 
2
 f
​
 
⋮
∂x 
n
​
 ∂x 
1
​
 
∂ 
2
 f
​
 
​
  
∂x 
1
​
 ∂x 
2
​
 
∂ 
2
 f
​
 
∂x 
2
2
​
 
∂ 
2
 f
​
 
⋮
∂x 
n
​
 ∂x 
2
​
 
∂ 
2
 f
​
 
​
  
⋯
⋯
⋱
⋯
​
  
∂x 
1
​
 ∂x 
n
​
 
∂ 
2
 f
​
 
∂x 
2
​
 ∂x 
n
​
 
∂ 
2
 f
​
 
⋮
∂x 
n
2
​
 
∂ 
2
 f
​
 
​
  
​
 
Symmetry
For functions with continuous second derivatives (Schwarz's theorem), the Hessian is symmetric: 
∂
2
f
∂
x
i
∂
x
j
=
∂
2
f
∂
x
j
∂
x
i
∂x 
i
​
 ∂x 
j
​
 
∂ 
2
 f
​
 = 
∂x 
j
​
 ∂x 
i
​
 
∂ 
2
 f
​
 

Numerical Computation with Finite Differences
When we don't have an analytical form, we can approximate the Hessian numerically:

Diagonal elements (second derivatives): 
∂
2
f
∂
x
i
2
≈
f
(
x
+
h
e
i
)
−
2
f
(
x
)
+
f
(
x
−
h
e
i
)
h
2
∂x 
i
2
​
 
∂ 
2
 f
​
 ≈ 
h 
2
 
f(x+he 
i
​
 )−2f(x)+f(x−he 
i
​
 )
​
 

Off-diagonal elements (mixed partials): 
∂
2
f
∂
x
i
∂
x
j
≈
f
(
x
+
h
e
i
+
h
e
j
)
−
f
(
x
+
h
e
i
−
h
e
j
)
−
f
(
x
−
h
e
i
+
h
e
j
)
+
f
(
x
−
h
e
i
−
h
e
j
)
4
h
2
∂x 
i
​
 ∂x 
j
​
 
∂ 
2
 f
​
 ≈ 
4h 
2
 
f(x+he 
i
​
 +he 
j
​
 )−f(x+he 
i
​
 −he 
j
​
 )−f(x−he 
i
​
 +he 
j
​
 )+f(x−he 
i
​
 −he 
j
​
 )
​
 

where 
e
i
e 
i
​
  is the unit vector in the 
i
i-th direction.

Example: Quadratic Function
For 
f
(
x
,
y
)
=
a
x
2
+
b
x
y
+
c
y
2
f(x,y)=ax 
2
 +bxy+cy 
2
 :

H
=
(
2
a
b
b
2
c
)
H=( 
2a
b
​
  
b
2c
​
 )

Applications in Machine Learning
Newton's Method: Uses the Hessian for second-order optimization: 
x
n
+
1
=
x
n
−
H
−
1
∇
f
x 
n+1
​
 =x 
n
​
 −H 
−1
 ∇f

Analyzing Critical Points:

If 
H
H is positive definite (all eigenvalues > 0): local minimum
If 
H
H is negative definite (all eigenvalues < 0): local maximum
If 
H
H has mixed signs: saddle point
Loss Surface Analysis: Understanding the curvature of neural network loss landscapes helps explain optimization dynamics.

Laplace Approximation: Uses the Hessian for Bayesian inference.

Computational Considerations
Computing the full Hessian requires 
O
(
n
2
)
O(n 
2
 ) function evaluations
For large neural networks, this is often impractical
Alternatives: Hessian-vector products, diagonal approximations, or quasi-Newton methods (BFGS, L-BFGS)

"""
from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Hessian matrix of function f at the given point using finite differences.
    
    Args:
        f: A scalar function that takes a list of floats and returns a float
        point: The point at which to compute the Hessian (list of coordinates)
        h: Step size for finite differences (default: 1e-5)
        
    Returns:
        The Hessian matrix as a list of lists (n x n where n = len(point))
    """
    n = len(point)
    hessian = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Compute diagonal elements (second derivatives)
    for i in range(n):
        # Create perturbed points
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[i] += h
        point_minus[i] -= h
        
        # Second derivative: [f(x+h) - 2f(x) + f(x-h)] / h^2
        hessian[i][i] = (f(point_plus) - 2 * f(point) + f(point_minus)) / (h ** 2)
    
    # Compute off-diagonal elements (mixed partial derivatives)
    for i in range(n):
        for j in range(i + 1, n):  # Only compute upper triangle, then use symmetry
            # Create the four perturbed points needed for mixed partials
            point_pp = point.copy()  # +h in i, +h in j
            point_pm = point.copy()  # +h in i, -h in j
            point_mp = point.copy()  # -h in i, +h in j
            point_mm = point.copy()  # -h in i, -h in j
            
            point_pp[i] += h
            point_pp[j] += h
            
            point_pm[i] += h
            point_pm[j] -= h
            
            point_mp[i] -= h
            point_mp[j] += h
            
            point_mm[i] -= h
            point_mm[j] -= h
            
            # Mixed partial: [f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)] / (4h^2)
            mixed_partial = (f(point_pp) - f(point_pm) - f(point_mp) + f(point_mm)) / (4 * h ** 2)
            
            # Use symmetry: ∂²f/∂xi∂xj = ∂²f/∂xj∂xi
            hessian[i][j] = mixed_partial
            hessian[j][i] = mixed_partial
    
    return hessian


