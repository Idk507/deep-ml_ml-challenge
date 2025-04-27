"""
Implement a function to calculate the Phi coefficient, a measure of the correlation between two binary variables. The function should take two lists of integers (0s and 1s) as input and return the Phi coefficient rounded to 4 decimal places.

Example:
Input:
phi_corr([1, 1, 0, 0], [0, 0, 1, 1])
Output:
-1.0
Reasoning:
The Phi coefficient measures the correlation between two binary variables. In this example, the variables have a perfect negative correlation, resulting in a Phi coefficient of -1.0.
The Phi coefficient
The Phi coefficient is a type of correlation coefficient , which is used when we need to find the correlation between two binary variables. For example when we have two variables x and y , x being gender and y signifying the presence of heart disease. Both variables are binary and if we need to find a correlation between x and y then we can use the formula below :

ϕ
=
(
x
00
⋅
x
11
)
−
(
x
01
⋅
x
10
)
(
x
00
+
x
01
)
(
x
10
+
x
11
)
(
x
00
+
x
10
)
(
x
01
+
x
11
)
ϕ= 
(x 
00
​
 +x 
01
​
 )(x 
10
​
 +x 
11
​
 )(x 
00
​
 +x 
10
​
 )(x 
01
​
 +x 
11
​
 )
​
 
(x 
00
​
 ⋅x 
11
​
 )−(x 
01
​
 ⋅x 
10
​
 )
​
 

Explanation of Terms:
(x_00): The number of cases where (x = 0) and (y = 0).
(x_01): The number of cases where (x = 0) and (y = 1).
(x_10): The number of cases where (x = 1) and (y = 0).
(x_11): The number of cases where (x = 1) and (y = 1).
"""
def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""
	import numpy as np
    x = np.array(x)
    y = np.array(y)
    x_00 = np.sum((x==0)&(y==0))
    x_01 = np.sum((x==0)&(y==1))
    x_10 = np.sum((x==1)&(y==0))
    x_11 = np.sum((x==1)&(y==1))
    val = (np.dot(x_00,x_11)-np.dot(x_01,x_10))/np.sqrt((x_00+x_01)*(x_10+x_11)*(x_00+x_10)*(x_01+x_11))
	return round(val,4)
