"""
Write a Python function to calculate the covariance matrix for a given set of vectors. The function should take a list of lists, where each inner list represents a feature with its observations, and return a covariance matrix as a list of lists. Additionally, provide test cases to verify the correctness of your implementation.

Example:
Input:
[[1, 2, 3], [4, 5, 6]]
Output:
[[1.0, 1.0], [1.0, 1.0]]
Reasoning:
The covariance between the two features is calculated based on their deviations from the mean. For the given vectors, both covariances are 1.0, resulting in a symmetric covariance matrix

Understanding Covariance Matrix
The covariance matrix is a fundamental concept in statistics and machine learning, used to understand the relationship between multiple variables (features) in a dataset. It quantifies the degree to which two variables change together.

Key Concepts
Covariance: Measures the directional relationship between two random variables. A positive covariance indicates that the variables increase together, while a negative covariance indicates that one variable increases as the other decreases.
Covariance Matrix: For a dataset with 
n
n features, the covariance matrix is an 
n
×
n
n×n matrix where each element 
(
i
,
j
)
(i,j) represents the covariance between the 
i
t
h
i 
th
  and 
j
t
h
j 
th
  features.
Covariance Formula
The covariance between two variables 
X
X and 
Y
Y is calculated as:

cov
(
X
,
Y
)
=
∑
k
=
1
m
(
X
k
−
X
ˉ
)
(
Y
k
−
Y
ˉ
)
m
−
1
cov(X,Y)= 
m−1
∑ 
k=1
m
​
 (X 
k
​
 − 
X
ˉ
 )(Y 
k
​
 − 
Y
ˉ
 )
​
 
Where:

X
k
X 
k
​
  and 
Y
k
Y 
k
​
  are the individual observations of variables 
X
X and 
Y
Y.
X
ˉ
X
ˉ
  and 
Y
ˉ
Y
ˉ
  are the means of 
X
X and 
Y
Y.
m
m is the number of observations.
Constructing the Covariance Matrix
Given a dataset with 
n
n features, the covariance matrix is constructed as follows:

Calculate the Mean: Compute the mean of each feature.
Compute Covariance: For each pair of features, calculate the covariance using the formula above.
Populate the Matrix: Place the computed covariance values in the corresponding positions in the matrix. The diagonal elements represent the variance of each feature.
Covariance Matrix
=
[
cov
(
X
1
,
X
1
)
cov
(
X
1
,
X
2
)
⋯
cov
(
X
1
,
X
n
)
cov
(
X
2
,
X
1
)
cov
(
X
2
,
X
2
)
⋯
cov
(
X
2
,
X
n
)
⋮
⋮
⋱
⋮
cov
(
X
n
,
X
1
)
cov
(
X
n
,
X
2
)
⋯
cov
(
X
n
,
X
n
)
]
Covariance Matrix= 
​
  
cov(X 
1
​
 ,X 
1
​
 )
cov(X 
2
​
 ,X 
1
​
 )
⋮
cov(X 
n
​
 ,X 
1
​
 )
​
  
cov(X 
1
​
 ,X 
2
​
 )
cov(X 
2
​
 ,X 
2
​
 )
⋮
cov(X 
n
​
 ,X 
2
​
 )
​
  
⋯
⋯
⋱
⋯
​
  
cov(X 
1
​
 ,X 
n
​
 )
cov(X 
2
​
 ,X 
n
​
 )
⋮
cov(X 
n
​
 ,X 
n
​
 )
​
  
​
 
Example Calculation
Consider the following dataset with two features:

Feature 1
:
[
1
,
2
,
3
]
Feature 2
:
[
4
,
5
,
6
]
Feature 1
Feature 2
​
  
:[1,2,3]
:[4,5,6]
​
 
Calculate Means:

X
ˉ
1
=
1
+
2
+
3
3
=
2.0
X
ˉ
2
=
4
+
5
+
6
3
=
5.0
X
ˉ
  
1
​
 = 
3
1+2+3
​
 =2.0
X
ˉ
  
2
​
 = 
3
4+5+6
​
 =5.0
Compute Covariances:

cov
(
X
1
,
X
1
)
=
(
1
−
2
)
2
+
(
2
−
2
)
2
+
(
3
−
2
)
2
3
−
1
=
1.0
cov
(
X
1
,
X
2
)
=
(
1
−
2
)
(
4
−
5
)
+
(
2
−
2
)
(
5
−
5
)
+
(
3
−
2
)
(
6
−
5
)
3
−
1
=
1.0
cov
(
X
2
,
X
2
)
=
(
4
−
5
)
2
+
(
5
−
5
)
2
+
(
6
−
5
)
2
3
−
1
=
1.0
cov(X 
1
​
 ,X 
1
​
 )= 
3−1
(1−2) 
2
 +(2−2) 
2
 +(3−2) 
2
 
​
 =1.0
cov(X 
1
​
 ,X 
2
​
 )= 
3−1
(1−2)(4−5)+(2−2)(5−5)+(3−2)(6−5)
​
 =1.0
cov(X 
2
​
 ,X 
2
​
 )= 
3−1
(4−5) 
2
 +(5−5) 
2
 +(6−5) 
2
 
​
 =1.0
Covariance Matrix:

[
1.0
1.0
1.0
1.0
]
[ 
1.0
1.0
​
  
1.0
1.0
​
 ]
Applications
Covariance matrices are widely used in various fields, including:

Principal Component Analysis (PCA): Reducing the dimensionality of datasets while preserving variance.
Portfolio Optimization: Understanding the variance and covariance between different financial assets.
Multivariate Statistics: Analyzing the relationships between multiple variables simultaneously.
Understanding the covariance matrix is crucial for interpreting the relationships in multivariate data and for performing advanced statistical analyses.
"""

import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_features = len(vectors)
    n_observations = len(vectors[0])
    covariance_matrix = [[0 for _ in range(n_features)] for _ in range(n_features)]

    means = [sum(feature) / n_observations for feature in vectors]

    for i in range(n_features):
        for j in range(i, n_features):
            covariance = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(n_observations)) / (n_observations - 1)
            covariance_matrix[i][j] = covariance_matrix[j][i] = covariance

    return covariance_matrix
