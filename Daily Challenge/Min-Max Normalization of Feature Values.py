"""
Implement a function that performs Min-Max Normalization on a list of integers, scaling all values to the range [0, 1]. Min-Max normalization helps ensure that all features contribute equally to a model by scaling them to a common range.

Example:
Input:
min_max([1, 2, 3, 4, 5])
Output:
[0.0, 0.25, 0.5, 0.75, 1.0]
Reasoning:
The minimum value is 1 and the maximum is 5. Each value is scaled using the formula (x - min) / (max - min).

Understanding Min-Max Normalization
Min-Max Normalization is a technique used to rescale numerical data to the range 
[
0
,
1
]
[0,1].

The formula used is:

X
′
=
X
−
X
min
⁡
X
max
⁡
−
X
min
⁡
X 
′
 = 
X 
max
​
 −X 
min
​
 
X−X 
min
​
 
​
 
Why Normalize?
Ensures all features have equal importance regardless of their original scale.
Commonly used in preprocessing for machine learning algorithms such as k-nearest neighbors, neural networks, and gradient descent-based models.
Special Case
If all the elements in the input are identical, then 
X
max
⁡
=
X
min
⁡
X 
max
​
 =X 
min
​
 . In that case, return an array of zeros.

Example
Given the input list [1, 2, 3, 4, 5]:

Minimum: 
1
1
Maximum: 
5
5
The normalized values are:
1
−
1
4
=
0.0
2
−
1
4
=
0.25
3
−
1
4
=
0.5
4
−
1
4
=
0.75
5
−
1
4
=
1.0
​
  
4
1−1
​
 =0.0
4
2−1
​
 =0.25
4
3−1
​
 =0.5
4
4−1
​
 =0.75
4
5−1
​
 =1.0
​
 
The result is [0.0, 0.25, 0.5, 0.75, 1.0].

Remember to round the result to 4 decimal places.

""""

import numpy as np
def min_max(x: list[int]) -> list[float]:
	# Your code here
    x = np.array(x)
    x_min = np.min(x)
    x_max = np.max(x)
    if x_min == x_max :
        return np.zeros_like(x)
    x_min_max =( x - x_min )/ (x_max - x_min)
    return x_min_max
