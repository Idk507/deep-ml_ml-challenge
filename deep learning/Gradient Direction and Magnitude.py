"""
Implement a function that calculates the magnitude and direction of a gradient vector. Given a gradient vector (which could represent the gradient of a loss function with respect to parameters), compute:

Magnitude: The L2 norm of the gradient vector, indicating how steep the function is at that point
Direction: The unit vector pointing in the direction of steepest ascent
Descent Direction: The unit vector pointing in the direction of steepest descent (used in gradient descent optimization)
The function should handle the edge case where the gradient is a zero vector (indicating a critical point). In this case, both direction vectors should be zero vectors.

Return a dictionary containing 'magnitude' (float), 'direction' (list), and 'descent_direction' (list).

Example:
Input:
gradient = [3.0, 4.0]
Output:
{'magnitude': 5.0, 'direction': [0.6, 0.8], 'descent_direction': [-0.6, -0.8]}
Reasoning:
The gradient vector is [3, 4]. The magnitude is sqrt(3^2 + 4^2) = sqrt(25) = 5.0. The direction (unit vector) is [3/5, 4/5] = [0.6, 0.8], pointing in the direction of steepest ascent. The descent direction is the negation: [-0.6, -0.8], which is the direction used in gradient descent optimization.

Learn About topic
Understanding Gradient Direction and Magnitude
What is a Gradient?
In machine learning, the gradient of a function 
f
(
x
)
f(x) is a vector of partial derivatives:

∇
f
(
x
)
=
[
∂
f
∂
x
1
,
∂
f
∂
x
2
,
…
,
∂
f
∂
x
n
]
∇f(x)=[ 
∂x 
1
​
 
∂f
​
 , 
∂x 
2
​
 
∂f
​
 ,…, 
∂x 
n
​
 
∂f
​
 ]

The gradient points in the direction of steepest increase of the function.

Gradient Magnitude
The magnitude (or norm) of the gradient measures how steep the function is at a given point. It is calculated as the L2 norm:

∥
∇
f
∥
=
∑
i
=
1
n
(
∂
f
∂
x
i
)
2
∥∇f∥= 
∑ 
i=1
n
​
 ( 
∂x 
i
​
 
∂f
​
 ) 
2
 
​
 

A larger magnitude indicates a steeper slope, while a magnitude of zero indicates a critical point (local minimum, maximum, or saddle point).

Gradient Direction
The direction of the gradient is represented by the unit vector:

d
^
=
∇
f
∥
∇
f
∥
d
^
 = 
∥∇f∥
∇f
​
 

This unit vector has magnitude 1 and points in the direction of steepest ascent.

Steepest Descent Direction
In optimization, we typically want to minimize a loss function. The direction of steepest descent is the opposite of the gradient direction:

d
^
descent
=
−
∇
f
∥
∇
f
∥
d
^
  
descent
​
 =− 
∥∇f∥
∇f
​
 

Application in Gradient Descent
In gradient descent, we update parameters by moving in the direction of steepest descent:

x
t
+
1
=
x
t
−
η
∇
f
(
x
t
)
x 
t+1
​
 =x 
t
​
 −η∇f(x 
t
​
 )

where 
η
η is the learning rate.

Understanding the gradient's magnitude helps with:

Learning rate selection: Large gradients might need smaller learning rates to avoid overshooting
Gradient clipping: Preventing exploding gradients by limiting magnitude
Convergence analysis: Near optimal points, magnitude approaches zero
Example Calculation
Given gradient 
g
=
[
3
,
4
]
g=[3,4]:

Magnitude: 
∥
g
∥
=
3
2
+
4
2
=
25
=
5
∥g∥= 
3 
2
 +4 
2
 
​
 = 
25
​
 =5

Direction: 
d
^
=
[
3
,
4
]
5
=
[
0.6
,
0.8
]
d
^
 = 
5
[3,4]
​
 =[0.6,0.8]

Descent direction: 
d
^
descent
=
[
−
0.6
,
−
0.8
]
d
^
  
descent
​
 =[−0.6,−0.8]

This tells us the function is relatively steep (magnitude 5) and increases most rapidly in the direction 
[
0.6
,
0.8
]
[0.6,0.8]. To minimize the function, we would move in the direction 
[
−
0.6
,
−
0.8
]
[−0.6,−0.8].

Edge Case: Zero Gradient
When 
∥
∇
f
∥
=
0
∥∇f∥=0, we are at a critical point. In this case, there is no well-defined direction, so both direction vectors are set to zero vectors.


"""

import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    
    Args:
        gradient: A list representing the gradient vector
    
    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient
        - direction: Unit vector in direction of steepest ascent
        - descent_direction: Unit vector in direction of steepest descent
    """
    # Convert to numpy array for easier computation
    grad_array = np.array(gradient, dtype=np.float64)
    
    # Calculate magnitude (L2 norm)
    magnitude = np.linalg.norm(grad_array)
    
    # Handle edge case: zero gradient (critical point)
    if magnitude == 0:
        # No well-defined direction at critical point
        zero_vector = [0.0] * len(gradient)
        return {
            'magnitude': 0.0,
            'direction': zero_vector,
            'descent_direction': zero_vector
        }
    
    # Calculate direction (unit vector in direction of steepest ascent)
    direction_array = grad_array / magnitude
    
    # Calculate descent direction (negative of direction)
    descent_direction_array = -direction_array
    
    # Convert back to lists and return
    return {
        'magnitude': float(magnitude),
        'direction': direction_array.tolist(),
        'descent_direction': descent_direction_array.tolist()
    }
