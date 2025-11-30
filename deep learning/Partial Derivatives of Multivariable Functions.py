"""
Partial Derivatives of Multivariable Functions
Medium
Calculus

Implement a function to compute partial derivatives of multivariable functions at a given point. Partial derivatives measure the rate of change with respect to one variable while holding others constant. Given a function name and a point, return the tuple of all partial derivatives at that point.

Example:
Input:
func_name='poly2d', point=(2.0, 3.0)
Output:
(21.0, 16.0)
Reasoning:
f(x,y) = x²y + xy². ∂f/∂x = 2xy + y² = 2(2)(3) + 9 = 21. ∂f/∂y = x² + 2xy = 4 + 2(2)(3) = 16. Gradient at (2,3) is (21, 16).

Learn About topic
Partial Derivatives
Partial derivatives measure the rate of change of a multivariable function with respect to one variable while holding others constant.

Definition
For 
f
(
x
,
y
)
f(x,y), the partial derivatives are:

∂
f
∂
x
=
lim
⁡
h
→
0
f
(
x
+
h
,
y
)
−
f
(
x
,
y
)
h
∂x
∂f
​
 = 
h→0
lim
​
  
h
f(x+h,y)−f(x,y)
​
 
∂
f
∂
y
=
lim
⁡
h
→
0
f
(
x
,
y
+
h
)
−
f
(
x
,
y
)
h
∂y
∂f
​
 = 
h→0
lim
​
  
h
f(x,y+h)−f(x,y)
​
 
Key idea: Treat other variables as constants.

Notation
Multiple notations for partial derivatives:

∂
f
∂
x
=
∂
x
f
=
f
x
=
D
x
f
∂x
∂f
​
 =∂ 
x
​
 f=f 
x
​
 =D 
x
​
 f
Computing Partial Derivatives
Rule: To compute 
∂
f
/
∂
x
∂f/∂x, treat all other variables as constants and differentiate normally.

Example: 
f
(
x
,
y
)
=
x
2
y
+
x
y
2
f(x,y)=x 
2
 y+xy 
2
 

For 
∂
f
/
∂
x
∂f/∂x (treat 
y
y as constant):

∂
f
∂
x
=
2
x
y
+
y
2
∂x
∂f
​
 =2xy+y 
2
 
For 
∂
f
/
∂
y
∂f/∂y (treat 
x
x as constant):

∂
f
∂
y
=
x
2
+
2
x
y
∂y
∂f
​
 =x 
2
 +2xy
Calculation Steps
Function: 
f
(
x
,
y
)
=
x
2
y
+
x
y
2
f(x,y)=x 
2
 y+xy 
2
  at 
(
2
,
3
)
(2,3)

Step 1: Compute 
∂
f
/
∂
x
∂f/∂x

∂
f
∂
x
=
2
x
y
+
y
2
∂x
∂f
​
 =2xy+y 
2
 
Step 2: Evaluate at 
(
2
,
3
)
(2,3)

∂
f
∂
x
∣
(
2
,
3
)
=
2
(
2
)
(
3
)
+
3
2
=
12
+
9
=
21
∂x
∂f
​
  
​
  
(2,3)
​
 =2(2)(3)+3 
2
 =12+9=21
Step 3: Compute 
∂
f
/
∂
y
∂f/∂y

∂
f
∂
y
=
x
2
+
2
x
y
∂y
∂f
​
 =x 
2
 +2xy
Step 4: Evaluate at 
(
2
,
3
)
(2,3)

∂
f
∂
y
∣
(
2
,
3
)
=
2
2
+
2
(
2
)
(
3
)
=
4
+
12
=
16
∂y
∂f
​
  
​
  
(2,3)
​
 =2 
2
 +2(2)(3)=4+12=16
Result: 
∇
f
(
2
,
3
)
=
(
21
,
16
)
∇f(2,3)=(21,16)

The Gradient
The gradient is the vector of all partial derivatives:

∇
f
=
(
∂
f
∂
x
,
∂
f
∂
y
,
∂
f
∂
z
,
…
)
∇f=( 
∂x
∂f
​
 , 
∂y
∂f
​
 , 
∂z
∂f
​
 ,…)
Properties:

Points in direction of steepest ascent
Magnitude = rate of steepest increase
Perpendicular to level curves/surfaces
Common Patterns
Power rule: 
f
(
x
,
y
)
=
x
n
y
m
f(x,y)=x 
n
 y 
m
 

∂
f
∂
x
=
n
x
n
−
1
y
m
,
∂
f
∂
y
=
m
x
n
y
m
−
1
∂x
∂f
​
 =nx 
n−1
 y 
m
 , 
∂y
∂f
​
 =mx 
n
 y 
m−1
 
Product rule: 
f
(
x
,
y
)
=
g
(
x
)
h
(
y
)
f(x,y)=g(x)h(y)

∂
f
∂
x
=
g
′
(
x
)
h
(
y
)
,
∂
f
∂
y
=
g
(
x
)
h
′
(
y
)
∂x
∂f
​
 =g 
′
 (x)h(y), 
∂y
∂f
​
 =g(x)h 
′
 (y)
Exponential: 
f
(
x
,
y
)
=
e
x
+
y
f(x,y)=e 
x+y
 

∂
f
∂
x
=
e
x
+
y
,
∂
f
∂
y
=
e
x
+
y
∂x
∂f
​
 =e 
x+y
 , 
∂y
∂f
​
 =e 
x+y
 
Chain rule: 
f
(
x
,
y
)
=
g
(
h
(
x
,
y
)
)
f(x,y)=g(h(x,y))

∂
f
∂
x
=
g
′
(
h
(
x
,
y
)
)
⋅
∂
h
∂
x
∂x
∂f
​
 =g 
′
 (h(x,y))⋅ 
∂x
∂h
​
 
Three or More Variables
For 
f
(
x
,
y
,
z
)
=
x
2
y
+
y
z
2
f(x,y,z)=x 
2
 y+yz 
2
 :

∂
f
∂
x
=
2
x
y
∂
f
∂
y
=
x
2
+
z
2
∂
f
∂
z
=
2
y
z
∂x
∂f
​
 
∂y
∂f
​
 
∂z
∂f
​
 
​
  
=2xy
=x 
2
 +z 
2
 
=2yz
​
  
​
 
At 
(
1
,
2
,
3
)
(1,2,3): 
∇
f
=
(
4
,
10
,
12
)
∇f=(4,10,12)

Geometric Interpretation
∂
f
/
∂
x
∂f/∂x at 
(
x
0
,
y
0
)
(x 
0
​
 ,y 
0
​
 ):

Slope of curve formed by slicing surface 
z
=
f
(
x
,
y
)
z=f(x,y) with plane 
y
=
y
0
y=y 
0
​
 
Rate of change moving in 
x
x-direction
∂
f
/
∂
y
∂f/∂y at 
(
x
0
,
y
0
)
(x 
0
​
 ,y 
0
​
 ):

Slope of curve formed by slicing surface with plane 
x
=
x
0
x=x 
0
​
 
Rate of change moving in 
y
y-direction
Application: Gradient Descent
To minimize 
f
(
x
,
y
)
f(x,y), update:

x
new
=
x
old
−
α
∂
f
∂
x
y
new
=
y
old
−
α
∂
f
∂
y
x 
new
​
 
y 
new
​
 
​
  
=x 
old
​
 −α 
∂x
∂f
​
 
=y 
old
​
 −α 
∂y
∂f
​
 
​
  
​
 
Where 
α
α is the learning rate.

Example: Minimize 
f
(
x
,
y
)
=
(
x
−
y
)
2
f(x,y)=(x−y) 
2
 

Partial derivatives:

∂
f
∂
x
=
2
(
x
−
y
)
,
∂
f
∂
y
=
−
2
(
x
−
y
)
∂x
∂f
​
 =2(x−y), 
∂y
∂f
​
 =−2(x−y)
At 
(
5
,
3
)
(5,3): 
∇
f
=
(
4
,
−
4
)
∇f=(4,−4)

With 
α
=
0.1
α=0.1:

x
new
=
5
−
0.1
(
4
)
=
4.6
y
new
=
3
−
0.1
(
−
4
)
=
3.4
x 
new
​
 
y 
new
​
 
​
  
=5−0.1(4)=4.6
=3−0.1(−4)=3.4
​
  
​
 
Moving toward minimum at 
x
=
y
x=y.

Higher-Order Partial Derivatives
Second-order partials:

∂
2
f
∂
x
2
,
∂
2
f
∂
y
2
,
∂
2
f
∂
x
∂
y
∂x 
2
 
∂ 
2
 f
​
 , 
∂y 
2
 
∂ 
2
 f
​
 , 
∂x∂y
∂ 
2
 f
​
 
Clairaut's Theorem: If continuous,

∂
2
f
∂
x
∂
y
=
∂
2
f
∂
y
∂
x
∂x∂y
∂ 
2
 f
​
 = 
∂y∂x
∂ 
2
 f
​
 
Mixed partials are equal (order doesn't matter).

Machine Learning Applications
Loss function: 
L
(
w
,
b
)
=
1
n
∑
(
y
i
−
(
w
x
i
+
b
)
)
2
L(w,b)= 
n
1
​
 ∑(y 
i
​
 −(wx 
i
​
 +b)) 
2
 

Gradient:

∂
L
∂
w
=
−
2
n
∑
x
i
(
y
i
−
(
w
x
i
+
b
)
)
∂w
∂L
​
 =− 
n
2
​
 ∑x 
i
​
 (y 
i
​
 −(wx 
i
​
 +b))
∂
L
∂
b
=
−
2
n
∑
(
y
i
−
(
w
x
i
+
b
)
)
∂b
∂L
​
 =− 
n
2
​
 ∑(y 
i
​
 −(wx 
i
​
 +b))
Used to update weights in gradient descent.

Neural networks: Backpropagation computes 
∂
L
/
∂
w
i
∂L/∂w 
i
​
  for all weights.

Key Differences from Single-Variable Calculus
Single variable: One derivative 
f
′
(
x
)
f 
′
 (x)

Multiple variables: Multiple partial derivatives, one per variable

Direction matters: Change depends on which direction you move

Gradient: Combines all partials into a vector


"""

import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    """
    Compute partial derivatives of multivariable functions at a given point.
    
    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x²y + xy²
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x·sin(y)
            'poly3d': f(x,y,z) = x²y + yz²
            'squared_error': f(x,y) = (x-y)²
        point: Point (x, y) or (x, y, z) at which to evaluate
    
    Returns:
        Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
    """
    if func_name == "poly2d":
        x, y = point
        dfdx = 2*x*y + y**2
        dfdy = x**2 + 2*x*y
        return (dfdx, dfdy)
    
    elif func_name == "exp_sum":
        x, y = point
        val = np.exp(x + y)
        return (val, val)
    
    elif func_name == "product_sin":
        x, y = point
        dfdx = np.sin(y)
        dfdy = x * np.cos(y)
        return (dfdx, dfdy)
    
    elif func_name == "poly3d":
        x, y, z = point
        dfdx = 2*x*y
        dfdy = x**2 + z**2
        dfdz = 2*y*z
        return (dfdx, dfdy, dfdz)
    
    elif func_name == "squared_error":
        x, y = point
        dfdx = 2*(x - y)
        dfdy = -2*(x - y)
        return (dfdx, dfdy)
    
    else:
        raise ValueError(f"Unknown function name: {func_name}")

