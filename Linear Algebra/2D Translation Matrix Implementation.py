"""
Task: Implement a 2D Translation Matrix
Your task is to implement a function that applies a 2D translation matrix to a set of points. A translation matrix is used to move points in 2D space by a specified distance in the x and y directions.

Write a function translate_object(points, tx, ty) where points is a list of [x, y] coordinates and tx and ty are the translation distances in the x and y directions, respectively.

The function should return a new list of points after applying the translation matrix.

Example:
Input:
points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3

print(translate_object(points, tx, ty))
Output:
[[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
Reasoning:
The translation matrix moves the points by 2 units in the x-direction and 3 units in the y-direction. The resulting points are [[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]].

2D Translation Matrix Implementation
The translation matrix is a fundamental concept in linear algebra and computer graphics, used to move points or objects in a 2D space.

Concept Overview
For a 2D translation, we use a 3x3 matrix to move a point ( (x, y) ) by ( x_t ) units in the x-direction and ( y_t ) units in the y-direction.

Any point ( P ) in 2D Cartesian space with coordinates ( (x, y) ) can be represented in homogeneous coordinates as ( (x, y, 1) ):

P
Cartesian
=
(
x
,
y
)
→
P
Homogeneous
=
(
x
,
y
,
1
)
P 
Cartesian
​
 =(x,y)→P 
Homogeneous
​
 =(x,y,1)
More generally, any scalar multiple of ( (x, y, 1) ) represents the same point in 2D space. Thus, ( (kx, ky, k) ) for any non-zero ( k ) also represents the same point ( (x, y) ).

The addition of this third coordinate allows us to represent translation as a linear transformation.

Translation Matrix
The translation matrix ( T ) is defined as:

T
=
[
1
0
x
t
0
1
y
t
0
0
1
]
T= 
​
  
1
0
0
​
  
0
1
0
​
  
x 
t
​
 
y 
t
​
 
1
​
  
​
 
Applying the Translation
To translate a point ( (x, y) ), we first convert it to homogeneous coordinates: ( (x, y, 1) ). The transformation is then performed using matrix multiplication:

[
1
0
x
t
0
1
y
t
0
0
1
]
[
x
y
1
]
=
[
x
+
x
t
y
+
y
t
1
]
​
  
1
0
0
​
  
0
1
0
​
  
x 
t
​
 
y 
t
​
 
1
​
  
​
  
​
  
x
y
1
​
  
​
 = 
​
  
x+x 
t
​
 
y+y 
t
​
 
1
​
  
​
 
Explanation of Parameters
Original Point: ( (x, y) )
Translation in x-direction: ( x_t )
Translation in y-direction: ( y_t )
Translated Point: ( (x + x_t, y + y_t) )
This process effectively shifts the original point ( (x, y) ) by ( x_t ) and ( y_t ), resulting in the new coordinates ( (x + x_t, y + y_t) ).

"""
import numpy as np

def translate_object(points, tx, ty):
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    
    homogeneous_points = np.hstack([np.array(points), np.ones((len(points), 1))])
    
    translated_points = np.dot(homogeneous_points, translation_matrix.T)
    
    return translated_points[:, :2].tolist()
