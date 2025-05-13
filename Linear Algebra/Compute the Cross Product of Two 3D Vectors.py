"""
Implement a function to compute the cross product of two 3-dimensional vectors. The cross product of two vectors results in a third vector that is perpendicular to both and follows the right-hand rule. This concept is fundamental in physics, engineering, and 3D graphics.

Example:
Input:
cross_product([1, 0, 0], [0, 1, 0])
Output:
[0, 0, 1]
Reasoning:
The cross product of two orthogonal unit vectors [1, 0, 0] and [0, 1, 0] is [0, 0, 1], pointing in the positive z-direction as per the right-hand rule.

Understanding the Cross Product
The cross product of two vectors 
a
⃗
a
  and 
b
⃗
b
  in 3D space is a vector that is perpendicular to both 
a
⃗
a
  and 
b
⃗
b
 .

Properties
Defined only in 3 dimensions.
The result 
c
⃗
=
a
⃗
×
b
⃗
c
 = 
a
 × 
b
  is perpendicular to both 
a
⃗
a
  and 
b
⃗
b
 .
Follows the right-hand rule.
Mathematical Formula
Given:

a
⃗
=
[
a
1
,
a
2
,
a
3
]
a
 =[a 
1
​
 ,a 
2
​
 ,a 
3
​
 ]
b
⃗
=
[
b
1
,
b
2
,
b
3
]
b
 =[b 
1
​
 ,b 
2
​
 ,b 
3
​
 ]
The cross product is:

a
⃗
×
b
⃗
=
[
a
2
b
3
−
a
3
b
2
,
 
a
3
b
1
−
a
1
b
3
,
 
a
1
b
2
−
a
2
b
1
]
a
 × 
b
 =[a 
2
​
 b 
3
​
 −a 
3
​
 b 
2
​
 , a 
3
​
 b 
1
​
 −a 
1
​
 b 
3
​
 , a 
1
​
 b 
2
​
 −a 
2
​
 b 
1
​
 ]
Use Cases
Calculating normals in 3D graphics.
Determining torque and angular momentum in physics.
Verifying orthogonality in machine learning geometry.
Example
For 
a
⃗
=
[
1
,
0
,
0
]
a
 =[1,0,0] and 
b
⃗
=
[
0
,
1
,
0
]
b
 =[0,1,0]:

a
⃗
×
b
⃗
=
[
0
,
0
,
1
]
a
 × 
b
 =[0,0,1]
The result points in the 
z
z-axis direction, confirming perpendicularity to both 
a
⃗
a
  and 
b
⃗
b
 .

"""
import numpy as np

def cross_product(a, b):
    # Your code here
    return np.cross(a,b)
