"""
Implement a function that solves a constrained quadratic optimization problem using the method of Lagrange multipliers.

Given a quadratic objective function and a linear equality constraint, find the optimal solution that minimizes the objective while satisfying the constraint.

The optimization problem is:

Minimize: f(x) = (1/2) x^T Q x + c^T x
Subject to: a^T x = b
Where:

Q is a 2x2 symmetric positive definite matrix (quadratic coefficients)
c is a 2-element vector (linear coefficients)
a is a 2-element vector (constraint coefficients)
b is a scalar (constraint right-hand side)
Your function should:

Set up the necessary conditions for optimality using Lagrange multipliers
Solve the resulting system of equations
Compute the objective function value at the optimal point
Return a dictionary containing:

'x': The optimal solution as a list of two values (rounded to 4 decimal places)
'lambda': The Lagrange multiplier value (rounded to 4 decimal places)
'objective': The objective function value at the optimum (rounded to 4 decimal places)
Note: The Lagrange multiplier represents the sensitivity of the optimal objective value to changes in the constraint.

Example:
Input:
Q = [[2, 0], [0, 2]], c = [0, 0], a = [1, 1], b = 2
Output:
{'x': [1.0, 1.0], 'lambda': 2.0, 'objective': 2.0}
Reasoning:
This minimizes f(x,y) = x^2 + y^2 subject to x + y = 2. The gradient of f is [2x, 2y] and the gradient of the constraint g(x,y) = x + y is [1, 1]. At the optimum, these must be parallel: [2x, 2y] = lambda * [1, 1]. Combined with x + y = 2, we get x = y = 1 and lambda = 2. The minimum objective value is 1^2 + 1^2 = 2.

Learn About topic
Lagrange Multipliers for Constrained Optimization
Lagrange multipliers provide a powerful method for finding extrema of functions subject to equality constraints. This technique is fundamental in machine learning, appearing in Support Vector Machines, constrained regression, and many other optimization problems.

The Core Idea
When optimizing 
f
(
x
)
f(x) subject to 
g
(
x
)
=
0
g(x)=0, the key insight is that at a constrained optimum, the gradient of 
f
f must be parallel to the gradient of 
g
g. If they weren't parallel, we could move along the constraint surface and still improve the objective.

Mathematically, this is expressed as:

∇
f
(
x
∗
)
=
λ
∇
g
(
x
∗
)
∇f(x 
∗
 )=λ∇g(x 
∗
 )

where 
λ
λ is the Lagrange multiplier.

The Lagrangian Function
We form the Lagrangian:

L
(
x
,
λ
)
=
f
(
x
)
−
λ
g
(
x
)
L(x,λ)=f(x)−λg(x)

The optimal point satisfies:

∇
x
L
=
0
and
∇
λ
L
=
0
∇ 
x
​
 L=0and∇ 
λ
​
 L=0

Quadratic Programming with Linear Constraints
For the quadratic objective 
f
(
x
)
=
1
2
x
T
Q
x
+
c
T
x
f(x)= 
2
1
​
 x 
T
 Qx+c 
T
 x with linear constraint 
a
T
x
=
b
a 
T
 x=b, the Lagrangian is:

L
(
x
,
λ
)
=
1
2
x
T
Q
x
+
c
T
x
−
λ
(
a
T
x
−
b
)
L(x,λ)= 
2
1
​
 x 
T
 Qx+c 
T
 x−λ(a 
T
 x−b)

Taking derivatives and setting to zero:

∇
x
L
=
Q
x
+
c
−
λ
a
=
0
∇ 
x
​
 L=Qx+c−λa=0 
∇
λ
L
=
−
(
a
T
x
−
b
)
=
0
∇ 
λ
​
 L=−(a 
T
 x−b)=0

This gives us the KKT (Karush-Kuhn-Tucker) system:

[
Q
−
a
a
T
0
]
[
x
λ
]
=
[
−
c
b
]
[ 
Q
a 
T
 
​
  
−a
0
​
 ][ 
x
λ
​
 ]=[ 
−c
b
​
 ]

Interpretation of the Lagrange Multiplier
The Lagrange multiplier 
λ
λ has an important economic interpretation: it represents the sensitivity of the optimal objective value to changes in the constraint. Specifically:

d
f
∗
d
b
=
λ
db
df 
∗
 
​
 =λ

If 
λ
>
0
λ>0, relaxing the constraint (increasing 
b
b) would improve the objective.

Applications in Machine Learning
Support Vector Machines: The SVM dual problem uses Lagrange multipliers to find support vectors
Maximum Entropy Models: Constraints on feature expectations lead to Lagrangian formulations
Regularized Regression: Can be viewed as constrained optimization with norm constraints
Portfolio Optimization: Minimize variance subject to expected return constraints

"""

import numpy as np

def lagrange_optimize(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> dict:
    """
    Solve constrained quadratic optimization using Lagrange multipliers.
    
    Minimize: f(x) = (1/2) x^T Q x + c^T x
    Subject to: a^T x = b
    
    Args:
        Q: 2x2 symmetric positive definite matrix
        c: 2-element vector (linear coefficients)
        a: 2-element vector (constraint coefficients)
        b: scalar (constraint value)
    
    Returns:
        Dictionary with 'x', 'lambda', and 'objective' keys
    """
    # Convert inputs to numpy arrays
    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    a = np.array(a, dtype=float)
    
    # Build the KKT system:
    # [Q  -a] [x]   = [-c]
    # [a^T 0] [λ]     [b]
    
    # Construct the coefficient matrix (3x3 for 2D problem)
    # Top rows: Q with -a as last column
    # Bottom row: a^T with 0 as last element
    KKT_matrix = np.zeros((3, 3))
    KKT_matrix[:2, :2] = Q  # Top-left: Q
    KKT_matrix[:2, 2] = -a  # Top-right: -a
    KKT_matrix[2, :2] = a   # Bottom-left: a^T
    KKT_matrix[2, 2] = 0    # Bottom-right: 0
    
    # Construct the right-hand side
    rhs = np.zeros(3)
    rhs[:2] = -c  # First two elements: -c
    rhs[2] = b    # Last element: b
    
    # Solve the linear system
    solution = np.linalg.solve(KKT_matrix, rhs)
    
    # Extract x and lambda
    x_opt = solution[:2]
    lambda_opt = solution[2]
    
    # Compute the objective function value at the optimum
    # f(x) = (1/2) x^T Q x + c^T x
    objective_value = 0.5 * np.dot(x_opt, np.dot(Q, x_opt)) + np.dot(c, x_opt)
    
    # Return results rounded to 4 decimal places
    return {
        'x': [round(float(x_opt[0]), 4), round(float(x_opt[1]), 4)],
        'lambda': round(float(lambda_opt), 4),
        'objective': round(float(objective_value), 4)
    }
