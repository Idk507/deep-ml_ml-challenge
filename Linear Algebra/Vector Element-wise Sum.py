"""
Write a Python function that computes the element-wise sum of two vectors. The function should return a new vector representing the resulting sum if the operation is valid, or -1 if the vectors have incompatible dimensions. Two vectors (lists) can be summed element-wise only if they are of the same length.

Example:
Input:
a = [1, 3], b = [4, 5]
Output:
[5, 8]
Reasoning:
Element-wise sum: [1+4, 3+5] = [5, 8].

Understanding Vector Element-wise Sum
In linear algebra, the element-wise sum (also known as vector addition) involves adding corresponding entries of two vectors.

Vector Notation
Given two vectors 
a
a and 
b
b of the same dimension 
n
n:

a
=
(
a
1
a
2
⋮
a
n
)
,
b
=
(
b
1
b
2
⋮
b
n
)
a= 
​
  
a 
1
​
 
a 
2
​
 
⋮
a 
n
​
 
​
  
​
 ,b= 
​
  
b 
1
​
 
b 
2
​
 
⋮
b 
n
​
 
​
  
​
 
The element-wise sum is defined as:

a
+
b
=
(
a
1
+
b
1
a
2
+
b
2
⋮
a
n
+
b
n
)
a+b= 
​
  
a 
1
​
 +b 
1
​
 
a 
2
​
 +b 
2
​
 
⋮
a 
n
​
 +b 
n
​
 
​
  
​
 
Key Requirement
Vectors 
a
a and 
b
b must be of the same length 
n
n for the operation to be valid. If their lengths differ, element-wise addition is not defined.

Example
Let:

a
=
[
1
,
2
,
3
]
,
b
=
[
4
,
5
,
6
]
a=[1,2,3],b=[4,5,6]
Then:

a
+
b
=
[
1
+
4
,
2
+
5
,
3
+
6
]
=
[
5
,
7
,
9
]
a+b=[1+4,2+5,3+6]=[5,7,9]
This simple operation is foundational in many applications such as vector arithmetic, neural network computations, and linear transformations.
"""
def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	if len(a) != len(b): return -1
	return [a[i] + b[i] for i in range(len(a))]
