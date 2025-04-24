"""
Write a Python function to perform a Phi Transformation that maps input features into a higher-dimensional space by generating polynomial features. The transformation allows models like linear regression to fit nonlinear data by introducing new feature dimensions that represent polynomial combinations of the original input features. The function should take a list of numerical data and a degree as inputs, and return a nested list where each inner list represents the transformed features of a data point. If the degree is less than 0, the function should return an empty list.

Example:
Input:
data = [1.0, 2.0], degree = 2
Output:
[[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]]
Reasoning:
The Phi Transformation generates polynomial features for each data point up to the specified degree. For data = [1.0, 2.0] and degree = 2, the transformation creates a nested list where each row contains powers of the data point from 0 to 2.

Phi Transformation
The Phi Transformation maps input features into a higher-dimensional space by generating polynomial features. This allows models like linear regression to fit nonlinear data by introducing new feature dimensions that represent polynomial combinations of the original input features.

Why Use Phi Transformation?
To increase the expressive power of simple models such as linear models.
To enable better fitting of nonlinear relationships in the data.
Equations
For an input value 
x
x, the Phi Transformation expands it as:

Φ
(
x
)
=
[
1
,
x
,
x
2
,
x
3
,
…
,
x
d
]
Φ(x)=[1,x,x 
2
 ,x 
3
 ,…,x 
d
 ]
Where 
d
d is the specified degree, and 
Φ
(
x
)
Φ(x) represents the transformed feature vector.

Example 1: Polynomial Expansion for One Value
Given 
x
=
3
x=3 and 
d
=
3
d=3, the Phi Transformation is:

Φ
(
3
)
=
[
1
,
3
,
9
,
27
]
Φ(3)=[1,3,9,27]
Example 2: Transformation for Multiple Values
For 
data
=
[
1
,
2
]
data=[1,2] and 
d
=
2
d=2, the Phi Transformation is:

Φ
(
[
1
,
2
]
)
=
[
1
1
1
1
2
4
]
Φ([1,2])=[ 
1
1
​
  
1
2
​
  
1
4
​
 ]
"""
import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	Returns:
		list[list[float]]: A nested list where each inner list represents the transformed features of a data point.
	"""
	if degree < 0 or not data:
		return []
	return np.array([[x ** i for i in range(degree + 1)] for x in data]).tolist()
