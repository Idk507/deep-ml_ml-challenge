"""
Write a Python function kernel_function that computes the linear kernel between two input vectors x1 and x2. The linear kernel is defined as the dot product (inner product) of two vectors.

Example:
Input:
import numpy as np

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

result = kernel_function(x1, x2)
print(result)
Output:
32
Reasoning:
The linear kernel between x1 and x2 is computed as:1*4 + 2*5 + 3*6 = 32

Understanding the Linear Kernel
A kernel function in machine learning is used to measure the similarity between two data points in a higher-dimensional space without having to compute the coordinates of the points in that space explicitly. The linear kernel is one of the simplest and most commonly used kernel functions. It computes the dot product (or inner product) of two vectors.

Mathematical Definition
The linear kernel between two vectors 
x
1
x 
1
​
  and 
x
2
x 
2
​
  is mathematically defined as:

K
(
x
1
,
x
2
)
=
x
1
⋅
x
2
=
∑
i
=
1
n
x
1
,
i
⋅
x
2
,
i
K(x 
1
​
 ,x 
2
​
 )=x 
1
​
 ⋅x 
2
​
 = 
i=1
∑
n
​
 x 
1,i
​
 ⋅x 
2,i
​
 
where 
n
n is the number of features, and 
x
1
,
i
x 
1,i
​
  and 
x
2
,
i
x 
2,i
​
  are the components of the vectors 
x
1
x 
1
​
  and 
x
2
x 
2
​
  respectively.

The linear kernel is widely used in support vector machines (SVMs) and other machine learning algorithms for linear classification and regression tasks. It is computationally efficient and works well when the data is linearly separable.

Characteristics
Simplicity: The linear kernel is straightforward to implement and compute.
Efficiency: It is computationally less expensive compared to other complex kernels like polynomial or RBF kernels.
Interpretability: The linear kernel is interpretable because it corresponds directly to the dot product, a well-understood operation in vector algebra.
In this problem, you will implement a function capable of computing the linear kernel between two vectors.
"""
import numpy as np

def kernel_function(x1, x2):
	# Your code here
    x1 ,x2 = np.array(x1),np.array(x2)
	return np.matmul(x1,x2)
